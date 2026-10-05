"""Evidence for the real-bag input fixes of 2026-09-28: what changed when
real scans started reaching PointNet++ in simulator order and padded to the
training point count (ros2_inference/_shared_inference.py, real_scan_to_points).

    python real_eval/before_after.py sim    # labelled simulated test split
    python real_eval/before_after.py bags   # the three real bags, label-free

`sim` answers it WITH ground truth: each test scan is also scored the way a
real Ouster would have delivered it -- in the Ouster's ring/column order
(sensor.ouster_to_sim_order is its own inverse), optionally with the
measured per-ring dropout -- through the old input path and the new one.

`bags` answers it WITHOUT ground truth on the real recordings, so its
numbers measure how coherent the predictions are, never whether they are
right.

LEGACY is the pre-fix input path, reproduced here and nowhere else: every
2nd column of the grid in the order it arrives, no-returns dropped, no
padding.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ros2_inference"))

import models.pointnet2_utils as pu  # noqa: E402
from _shared_inference import real_scan_to_points  # noqa: E402
from dataset import load_grid, prepare_points, resolve_recording_path  # noqa: E402
from evaluate import apply_threshold, score  # noqa: E402
from models.pointnet2 import NUM_CLASSES, PointNet2MultiTask  # noqa: E402
from real_noise_augment import REAL_RING_DROPOUT_RATE, inject_real_sensor_noise  # noqa: E402
from sensor import N_COLS, N_RINGS, ouster_to_sim_order  # noqa: E402

CHECKPOINT = REPO / "outputs" / "pointnet2_checkpoints" / "epoch39_multitask_model.pt"
THRESHOLDS = (0.97, 0.0)
_LEGACY_IDX = np.arange(N_RINGS * N_COLS).reshape(N_RINGS, N_COLS)[:, ::2].reshape(-1)

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def legacy_points(xyz_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pre-fix input path, same signature as real_scan_to_points."""
    flat = xyz_grid[:, ::2].reshape(-1, 3)
    kept = np.flatnonzero(np.isfinite(flat).all(axis=-1))
    return flat[kept].astype(np.float32), _LEGACY_IDX[kept]


def load_model(device):
    model = PointNet2MultiTask().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    return model.eval()


@torch.no_grad()
def classify(model, pts, device, variable_n):
    """-> (argmax, confidence) per input point. A variable point count runs
    FPS without CUDA graphs, whose cache keeps one capture per count."""
    pu._fps_graph_disabled = variable_n
    logits = model(torch.from_numpy(pts).unsqueeze(0).to(device))["class_logits"][0]
    conf, pred = logits.softmax(dim=-1).max(dim=-1)
    return pred, conf


def run_sim(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    rec = REPO / "scenario_data" / "mbut_76scenarios" / "o_robot_nav"
    test_ids = np.load(REPO / "data" / "sim" / "test_ids.npy")[:args.limit]
    conditions = {
        "A": "simulator order, training input path",
        "B": "Ouster order, old input path",
        "C": "Ouster order + real dropout, old input path",
        "D": "Ouster order + real dropout, new input path",
    }
    confusion = {(c, t): torch.zeros(NUM_CLASSES ** 2, dtype=torch.int64)
                 for c in conditions for t in THRESHOLDS}

    def add(cond, pred, conf, labels):
        lab = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(device)
        for t in THRESHOLDS:
            idx = (lab * NUM_CLASSES + apply_threshold(pred, conf, t)).cpu()
            confusion[cond, t] += torch.bincount(idx, minlength=NUM_CLASSES ** 2)

    for i, sid in enumerate(test_ids):
        path = resolve_recording_path(rec, str(sid))
        xyz, cls = load_grid(path)
        seed = zlib.crc32(path.name.encode("utf-8"))

        # A: exactly evaluate.py's input, padding duplicates included, so it
        # must reproduce the published macro-F1.
        pts, lab, _ = prepare_points(xyz, cls, pad_seed=seed)
        add("A", *classify(model, pts, device, False), lab)

        ou_xyz, ou_cls = ouster_to_sim_order(xyz), ouster_to_sim_order(cls).reshape(-1)
        pts, idx = legacy_points(ou_xyz)
        add("B", *classify(model, pts, device, True), ou_cls[idx])

        # The measured profile is in Ouster row order, like this grid; objects
        # drop too, as they do on the real sensor. Same mask for C and D.
        noisy = inject_real_sensor_noise(ou_xyz, np.random.default_rng(seed),
                                         ring_dropout_rate=REAL_RING_DROPOUT_RATE)
        pts, idx = legacy_points(noisy)
        add("C", *classify(model, pts, device, True), ou_cls[idx])

        pts, idx = real_scan_to_points(noisy)
        pred, conf = classify(model, pts, device, False)
        n = len(idx)  # padding duplicates dropped, as the real-data runner does
        add("D", pred[:n], conf[:n], ou_cls[idx])

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test_ids)} scans", flush=True)

    results = []
    for c, desc in conditions.items():
        entry = {"condition": c, "description": desc}
        for t in THRESHOLDS:
            rows, macro, acc = score(confusion[c, t])
            entry[f"t{t:g}"] = {"macroF1": macro, "overallPointAcc": acc, "rows": rows}
        results.append(entry)
        print(f"{c}  {desc:<48} macro-F1 @0.97 {entry['t0.97']['macroF1']:.3f}   "
              f"@argmax {entry['t0']['macroF1']:.3f}")

    out = REPO / "outputs" / "measured" / "preprocessing_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"script": "real_eval/before_after.py sim", "checkpoint": str(CHECKPOINT),
                               "testSamples": int(len(test_ids)), "conditions": results}, indent=2),
                   encoding="utf-8")
    plot_sim(results, out.with_suffix(".png"))
    print(f"wrote {out} and {out.with_suffix('.png')}")


def plot_sim(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(results))
    w = 0.36
    for k, (t, color, name) in enumerate(((0.97, BLUE, "threshold 0.97 (used on real data)"),
                                          (0.0, ORANGE, "plain argmax"))):
        vals = [r[f"t{t:g}"]["macroF1"] for r in results]
        bars = ax.bar(x + (k - 0.5) * w, vals, w - 0.03, color=color, label=name, zorder=3)
        if t == 0.97:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"{r['condition']}\n{r['description'].replace(', ', chr(10))}" for r in results],
                  fontsize=8, color=INK2)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("macro-F1, object classes", fontsize=9, color=INK2)
    ax.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_title("PointNet++ (epoch39) on the simulated test split, scans fed as the real sensor delivers them",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


BAGS = {
    "bag1_20260910_144421": Path(r"E:\real_bags\bag1_20260910_144421\bag1_20260910_144421"),
    "bag2_20260910_151509": Path(r"E:\real_bags\bag1_20260910_144421\bag2_20260910_151509"),
    "bag3_20260910_152806": REPO / "real_bags" / "bag3_20260910_152806",
}
RUNS = REPO / "outputs" / "real_inference"
STRIDE = 4  # what detect_persistent_objects' MIN_SCANS is sized for


def bag_metrics(run_dir: Path, analysis: dict) -> dict:
    p = np.load(run_dir / "predictions.npz")
    use = p["scan_index"] % STRIDE == 0
    cl = analysis["clusters"]
    pts = sum(c["points"] for c in cl)
    return {
        "objectPointsPerScan": analysis["objectPointsPerScan"],
        "scatteredShare": analysis["scatteredShare"],
        "persistentShare": analysis["persistentShare"],
        "persistentClusters": len(cl),
        "clusterClassPurity": sum(c["classPurity"] * c["points"] for c in cl) / pts if pts else None,
        "meanReconMse": float(p["mse"][use].mean()),
    }


def run_bags(args):
    import _shared_inference
    from detect_persistent_objects import analyze
    from run_inference_on_bag import run_bag
    from _shared_inference import ModelRunner

    results = {}
    for name in args.bags:
        after_dir, before_dir = RUNS / name, RUNS / f"{name}_legacy"
        if not (after_dir / "predictions.npz").exists():
            raise SystemExit(f"{after_dir} missing -- run ros2_inference/run_inference_on_bag.py first")
        after = analyze(after_dir, STRIDE)
        yaw = np.radians(after["mountYawDeg"])  # a property of the robot, not of the model

        if not (before_dir / "predictions.npz").exists() or args.rerun:
            runner = ModelRunner("pointnet2", CHECKPOINT, 14.425209045410156, class_conf_threshold=0.97)
            real = _shared_inference.real_scan_to_points
            _shared_inference.real_scan_to_points = legacy_points
            pu._fps_graph_disabled = True  # variable point count: see classify()
            try:
                run_bag(BAGS[name], before_dir, runner, stride=STRIDE, video=False)
            finally:
                _shared_inference.real_scan_to_points = real
                pu._fps_graph_disabled = False
        before = analyze(before_dir, STRIDE, mount_yaw=yaw)
        results[name] = {"before": bag_metrics(before_dir, before), "after": bag_metrics(after_dir, after)}
        print(f"{name}: " + "; ".join(f"{k} {results[name]['before'][k]:.3g} -> {results[name]['after'][k]:.3g}"
                                      for k in results[name]["after"] if results[name]["after"][k] is not None))

    out = RUNS / "before_after.json"
    out.write_text(json.dumps({"script": "real_eval/before_after.py bags", "stride": STRIDE,
                               "note": "label-free: coherence of the predictions, not their correctness",
                               "bags": results}, indent=2), encoding="utf-8")
    plot_bags(results, RUNS / "before_after_metrics.png")
    compare_frames(BAGS["bag3_20260910_152806"], RUNS / "bag3_20260910_152806", RUNS / "before_after_frames.png")
    print(f"wrote {out}, before_after_metrics.png, before_after_frames.png")


def plot_bags(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("objectPointsPerScan", "object points per scan", "{:.0f}"),
               ("scatteredShare", "scattered share\n(isolated in their own scan)", "{:.0%}"),
               ("persistentShare", "persistent share\n(near field, >= 8 scans agree)", "{:.0%}"),
               ("persistentClusters", "persistent clusters", "{:.0f}")]
    names = list(results)
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3.4), dpi=150, facecolor=SURFACE)
    x = np.arange(len(names))
    for ax, (key, title, fmt) in zip(axes, metrics):
        ax.set_facecolor(SURFACE)
        for k, (when, color) in enumerate((("before", ORANGE), ("after", BLUE))):
            vals = [results[n][when][key] for n in names]
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.35, color=color, zorder=3,
                          label="before fix (old input path)" if when == "before" else "after fix")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom",
                        fontsize=6.5, color=INK2)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xticks(x, [n.split("_")[0] for n in names], fontsize=8, color=INK2)
        ax.tick_params(axis="y", labelsize=7, colors=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK2, loc="upper left")
    fig.suptitle("Real bags, every 4th scan, threshold 0.97 -- no ground truth: these measure how coherent "
                 "the predictions are, not whether they are right", fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def compare_frames(bag: Path, after_dir: Path, path: Path, n: int = 3):
    """The same real scans through the old and the new input path: measured
    range on top, then each path's predicted classes."""
    import _shared_inference
    from _shared_inference import ModelRunner
    from playback_common import class_rgb, label_row, range_rgb
    from run_inference_on_bag import read_bag
    from PIL import Image

    p = np.load(after_dir / "predictions.npz")
    obj = p["class_counts"][:, 4:].sum(axis=1)
    early = np.flatnonzero((p["scan_index"] % STRIDE == 0) & (p["scan_index"] < 1500))
    pick = sorted(p["scan_index"][early[np.argsort(-obj[early])[:n]]].tolist())

    grids = {}
    for i, (_k, _t, xyz) in enumerate(read_bag(bag, "/ouster/points")):
        if i in pick:
            grids[i] = xyz
        if i >= pick[-1]:
            break

    def class_row(runner, xyz):
        d = runner.run_detailed(xyz)
        row = d["image"][-N_RINGS:].copy()  # the predicted-class panel
        row[:14, :300] = class_rgb(np.zeros((1, 1), dtype=np.uint8))  # clear its label for ours
        return row, sum(v for k, v in d["pred_class_counts"].items() if k >= 4)

    new = ModelRunner("pointnet2", CHECKPOINT, 14.425209045410156)
    old = ModelRunner("pointnet2", CHECKPOINT, 14.425209045410156)
    gap = np.full((6, N_COLS, 3), 250, dtype=np.uint8)
    blocks = []
    for i in pick:
        after_row, n_after = class_row(new, grids[i])
        real = _shared_inference.real_scan_to_points
        _shared_inference.real_scan_to_points = legacy_points
        pu._fps_graph_disabled = True
        try:
            before_row, n_before = class_row(old, grids[i])
        finally:
            _shared_inference.real_scan_to_points = real
            pu._fps_graph_disabled = False
        rng = np.linalg.norm(grids[i], axis=-1)
        blocks += [label_row(range_rgb(rng), f"{bag.name} scan {i}: measured range"),
                   label_row(before_row, f"BEFORE (old input path): {n_before} object points"),
                   label_row(after_row, f"AFTER (fixed input path): {n_after} object points"), gap]
    img = np.concatenate(blocks[:-1])
    Image.fromarray(np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sim", help="ground-truth ablation on the simulated test split")
    s.add_argument("--limit", type=int, default=None, help="only the first N test scans (smoke runs)")
    b = sub.add_parser("bags", help="label-free before/after on the real bags")
    b.add_argument("--bags", nargs="+", default=list(BAGS), choices=list(BAGS))
    b.add_argument("--rerun", action="store_true", help="redo the old-path runs even if present")
    args = ap.parse_args()
    if args.mode == "sim":
        run_sim(args)
    else:
        run_bags(args)


if __name__ == "__main__":
    main()
