"""Run a trained encoder over every /ouster/points scan in a recorded ROS2
bag, offline -- no ROS2 install needed. Uses `rosbags` (pip install rosbags),
a pure-Python CDR deserializer that reads ROS2 bags (.db3/.mcap) directly
from their own self-described message schemas, without rclpy or any ROS2
runtime. See live_inference_node.py for the live-ROS2 version; both call the
same per-scan code in _shared_inference.py.

    python ros2_inference/run_inference_on_bag.py --bag real_bags/bag3_20260910_152806

writes outputs/real_inference/<bag name>/:
  inference.mp4    every scan at the bag's own rate (H.264, plays in a browser)
  predictions.npz  per scan: time, class counts, recon MSE; every point the
                   model's argmax put in an object class, with its confidence
                   (so any threshold can be applied later without the GPU);
                   the odometry; every 10th scan downsampled, for maps
                   (real_eval/detect_persistent_objects.py reads this file)
  summary.json     settings, scans used and skipped (by reason), class
                   totals, mean MSE, peak GPU memory, runtime

One pass over the bag, one scan in memory at a time. A scan that cannot be
used -- corrupt message, wrong shape, too few returns -- is skipped with its
reason and counted, not fatal; more than 1% skipped fails the run (exit code
1) after writing everything, so the outputs can be inspected. Anything else
(a model or CUDA error) still stops the run: that is a bug, not bad data.

Run over all three recorded bags on 2026-09-28 (bag1, bag2 on E:\\real_bags,
bag3 in real_bags/).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared_inference import N_COLS, N_RINGS, ModelRunner, SkipScan  # noqa: E402
from dataset import CLASS_NAMES, NUM_CLASSES, OBJECT_CLASSES  # noqa: E402
from playback_common import encode_frame, label_row, patch_template  # noqa: E402

DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "pointnet2_checkpoints" / "epoch39_multitask_model.pt"
ODOM_TOPIC = "/j100_0847/platform/odom/filtered"
MAX_SKIPPED_FRACTION = 0.01
MAP_EVERY = 10  # keep every 10th scan (1 s at 10 Hz), downsampled, for maps
FLOAT32 = 7     # sensor_msgs/PointField datatype


def pointcloud2_xyz(msg) -> np.ndarray:
    """PointCloud2 -> (height, width, 3) float32 x/y/z. Raises ValueError on
    a layout this reader doesn't handle, rather than misreading it."""
    fields = {f.name: f for f in msg.fields}
    for name in "xyz":
        if name not in fields or fields[name].datatype != FLOAT32:
            raise ValueError(f"field {name!r} missing or not float32")
    if msg.is_bigendian:
        raise ValueError("big-endian point cloud")
    if msg.row_step != msg.width * msg.point_step or len(msg.data) != msg.height * msg.row_step:
        raise ValueError(f"data length {len(msg.data)} doesn't match "
                         f"{msg.height}x{msg.width}x{msg.point_step}")
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, msg.point_step)
    return np.stack([buf[..., fields[a].offset:fields[a].offset + 4].copy().view(np.float32)[..., 0]
                     for a in "xyz"], axis=-1)


def _stamp_ns(msg) -> int:
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def read_bag(bag: Path, topic: str, odom_topic: str | None = None, info: dict | None = None):
    """Yields ("scan", t_ns, xyz_grid or the ValueError that made it
    unreadable) and ("odom", t_ns, (x, y, yaw)) in time order, one message in
    memory at a time.

    Times are the messages' header stamps when both topics stamp in ROS time
    (checked on each topic's first message), else the bag's record times --
    either way the two topics share one clock. `info`, if given, receives
    "n_scans", "clock" and "odom" (whether the odometry topic exists)."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        conns = {c.topic: c for c in reader.connections}
        if topic not in conns:
            raise SystemExit(f"topic {topic!r} not found in bag. Available topics:\n  "
                             + "\n  ".join(sorted(conns)))
        wanted = [conns[topic]] + ([conns[odom_topic]] if odom_topic in conns else [])

        use_header = True
        for conn in wanted:
            for c, t, raw in reader.messages(connections=[conn]):
                use_header &= abs(_stamp_ns(reader.deserialize(raw, c.msgtype)) - t) < 10**9
                break
        if info is not None:
            info.update(n_scans=conns[topic].msgcount, clock="header" if use_header else "record",
                        odom=odom_topic in conns)

        for c, t, raw in reader.messages(connections=wanted):
            try:
                msg = reader.deserialize(raw, c.msgtype)
            except Exception as e:  # a corrupt record is bad data, not a bug here
                if c.topic == topic:
                    yield "scan", t, ValueError(f"deserialize failed: {type(e).__name__}")
                continue
            if use_header:
                t = _stamp_ns(msg)
            if c.topic == topic:
                try:
                    yield "scan", t, pointcloud2_xyz(msg)
                except ValueError as e:
                    yield "scan", t, e
            else:
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                yield "odom", t, (p.x, p.y, yaw)


def classify_scan(runner: ModelRunner, xyz):
    """-> run_detailed()'s dict, or the reason (str) this scan is skipped."""
    if isinstance(xyz, ValueError):
        return f"unreadable message: {xyz}"
    if xyz.shape[:2] != (N_RINGS, N_COLS):
        return (f"shape {xyz.shape[0]}x{xyz.shape[1]}, not {N_RINGS}x{N_COLS} -- check the OS1's "
                f"channel count / horizontal resolution")
    try:
        return runner.run_detailed(xyz)
    except SkipScan as e:
        return f"skipped by the model path: {e}"


def map_points(xyz: np.ndarray) -> np.ndarray:
    """A light copy of one scan for top-down maps: 1-15 m away horizontally,
    from ~0.2 m above the floor to 1.5 m (walls and obstacles, not the floor
    itself), one point per 0.2 m voxel. Sensor frame."""
    p = xyz.reshape(-1, 3)
    h = np.hypot(p[:, 0], p[:, 1])
    with np.errstate(invalid="ignore"):
        p = p[(h > 1.0) & (h < 15.0) & (p[:, 2] > -0.4) & (p[:, 2] < 1.5)]
    _, first = np.unique(np.floor(p / 0.2).astype(np.int32), axis=0, return_index=True)
    return p[np.sort(first)]


def open_video(path: Path, w: int, h: int, fps: float):
    """H.264 through the ffmpeg that imageio_ffmpeg ships (the same one
    demo/build_demo.py uses) -- this machine's OpenCV has no H.264 encoder,
    and its mp4v fallback is a codec no browser plays."""
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(
        str(path), (w, h), fps=fps, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        quality=None, macro_block_size=2,
        output_params=["-crf", "20", "-preset", "medium", "-movflags", "+faststart"])
    writer.send(None)
    return writer


def status_bar(width: int, text: str) -> np.ndarray:
    return label_row(np.full((18, width, 3), 14, dtype=np.uint8), text)


def _even(img: np.ndarray) -> np.ndarray:
    """yuv420p needs even dimensions."""
    return np.pad(img, ((0, img.shape[0] % 2), (0, img.shape[1] % 2), (0, 0)))


def run_bag(bag: Path, out_dir: Path, runner: ModelRunner, *, topic="/ouster/points",
            odom_topic=ODOM_TOPIC, stride=1, limit=None, video=True, html: Path | None = None,
            max_frames=200) -> dict:
    """The whole offline run; returns the summary dict (also written to
    out_dir/summary.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info: dict = {}
    t_start = time.perf_counter()
    if runner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows = {k: [] for k in ("scan_index", "t_ns", "mse", "class_counts", "n_points")}
    obj = {k: [] for k in ("xyz", "pred", "conf")}
    obj_offsets = [0]
    odom = []
    map_rows, map_xyz, map_offsets = [], [], [0]
    skipped: dict[str, int] = {}
    writer = video_path = None
    html_frames, html_meta, html_pick = [], [], None
    i = -1

    for kind, t_ns, payload in read_bag(bag, topic, odom_topic, info):
        if kind == "odom":
            odom.append((t_ns, *payload))
            continue
        i += 1
        if i == 0:
            n_total = info["n_scans"]
            print(f"{bag.name}: {n_total} scans, timestamps from the {info['clock']} clock"
                  + ("" if info["odom"] else f"; no {odom_topic} in this bag -- no maps"), flush=True)
            if html is not None:
                html_pick = set(np.linspace(0, n_total - 1, min(max_frames, n_total)).round().astype(int).tolist())
        if limit is not None and len(rows["t_ns"]) >= limit:
            break
        if i % stride:
            continue

        d = classify_scan(runner, payload)
        if isinstance(d, str):
            skipped[d] = skipped.get(d, 0) + 1
            continue

        counts = np.zeros(NUM_CLASSES, dtype=np.int32)
        for c, n in (d["pred_class_counts"] or {}).items():  # None: model has no classifier
            counts[c] = n
        rows["scan_index"].append(i)
        rows["t_ns"].append(t_ns)
        rows["mse"].append(d["mse"])
        rows["class_counts"].append(counts)
        if "pred" in d:
            is_obj = np.isin(d["pred"], OBJECT_CLASSES)
            obj["xyz"].append(d["points"][is_obj])
            obj["pred"].append(d["pred"][is_obj].astype(np.uint8))
            obj["conf"].append(d["conf"][is_obj].astype(np.float16))
            obj_offsets.append(obj_offsets[-1] + int(is_obj.sum()))
            rows["n_points"].append(len(d["pred"]))
        if len(rows["t_ns"]) % MAP_EVERY == 1:
            pts = map_points(payload)
            map_rows.append(len(rows["t_ns"]) - 1)
            map_xyz.append(pts)
            map_offsets.append(map_offsets[-1] + len(pts))

        t_s = (t_ns - rows["t_ns"][0]) / 1e9
        shown = " ".join(f"{CLASS_NAMES[c]} {counts[c]}" for c in OBJECT_CLASSES)
        frame = None
        if video:
            frame = _even(np.concatenate([status_bar(d["image"].shape[1],
                f"{bag.name}  t={t_s:6.1f}s  scan {i + 1}/{n_total}  recon mse {d['mse']:.4f}  "
                f"objects shown (conf >= {runner.class_conf_threshold:g}): {shown}"), d["image"]]))
            frame = np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)  # 64-row panels are thin
            if writer is None:
                video_path = out_dir / "inference.mp4"
                writer = open_video(video_path, frame.shape[1], frame.shape[0], 10.0 / stride)
            writer.send(np.ascontiguousarray(frame))
        if html_pick is not None and i in html_pick:
            html_frames.append(encode_frame(d["image"], upscale=2))
            html_meta.append({"t": t_s, "mse": d["mse"],
                              "class_counts": {str(c): int(n) for c, n in d["pred_class_counts"].items()}})

        n_done = len(rows["t_ns"])
        if n_done % 100 == 0:
            el = time.perf_counter() - t_start
            left = ((n_total - i - 1) / stride) * el / n_done
            print(f"  {n_done} scans in {el / 60:.1f} min, ~{left / 60:.0f} min left", flush=True)

    if writer is not None:
        writer.close()
    if i < 0:
        raise SystemExit(f"no messages on {topic}")

    n_used = len(rows["t_ns"])
    n_skipped = sum(skipped.values())
    np.savez_compressed(
        out_dir / "predictions.npz",
        scan_index=np.array(rows["scan_index"], dtype=np.int32),
        t_ns=np.array(rows["t_ns"], dtype=np.int64),
        mse=np.array(rows["mse"], dtype=np.float32),
        class_counts=np.array(rows["class_counts"], dtype=np.int32).reshape(-1, NUM_CLASSES),
        n_points=np.array(rows["n_points"], dtype=np.int32),
        obj_offsets=np.array(obj_offsets, dtype=np.int64),
        obj_xyz=np.concatenate(obj["xyz"]) if obj["xyz"] else np.zeros((0, 3), np.float32),
        obj_pred=np.concatenate(obj["pred"]) if obj["pred"] else np.zeros(0, np.uint8),
        obj_conf=np.concatenate(obj["conf"]) if obj["conf"] else np.zeros(0, np.float16),
        odom=np.array(odom, dtype=np.float64).reshape(-1, 4),
        map_row=np.array(map_rows, dtype=np.int32),
        map_offsets=np.array(map_offsets, dtype=np.int64),
        map_xyz=np.concatenate(map_xyz) if map_xyz else np.zeros((0, 3), np.float32),
    )
    if html is not None and html_frames:
        write_html(html, html_frames, html_meta, bag, runner)

    totals = np.array(rows["class_counts"], dtype=np.int64).reshape(-1, NUM_CLASSES).sum(axis=0)
    summary = {
        "bag": str(bag), "model": runner.model_name, "classThreshold": runner.class_conf_threshold,
        "clock": info.get("clock"), "scansInBag": info.get("n_scans"), "stride": stride,
        "scansUsed": n_used, "scansSkipped": n_skipped, "skippedByReason": skipped,
        "skippedFraction": n_skipped / max(n_used + n_skipped, 1),
        "status": "ok" if n_skipped <= MAX_SKIPPED_FRACTION * (n_used + n_skipped) else "too many skipped",
        "predictedPointsByClass": {CLASS_NAMES[c]: int(totals[c]) for c in range(NUM_CLASSES) if totals[c]},
        "meanObjectPointsPerScan": float(totals[list(OBJECT_CLASSES)].sum() / max(n_used, 1)),
        "meanReconMse": float(np.mean(rows["mse"])) if n_used else None,
        "odometryMessages": len(odom),
        "peakGpuMemoryMiB": (torch.cuda.max_memory_allocated() / 2**20
                             if runner.device.type == "cuda" else None),
        "runtimeMin": (time.perf_counter() - t_start) / 60,
        "video": str(video_path) if video_path else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{bag.name}: {n_used} scans used, {n_skipped} skipped "
          f"({summary['skippedFraction']:.2%}) in {summary['runtimeMin']:.1f} min")
    for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>6}  {reason}")
    print(f"  mean object points per scan (conf >= {runner.class_conf_threshold:g}): "
          f"{summary['meanObjectPointsPerScan']:.1f}; mean recon mse {summary['meanReconMse']}")
    print(f"  wrote {out_dir}")
    return summary


def bag_frames_payload(*, bag, topic, runner, max_frames=None, frame_sink=None,
                        upscale=1, progress_every=50):
    """Same {"frames", "meta", "stats"} shape build_playback_payload returns,
    for a real bag instead of a simulated scenario -- so one demo exporter and
    one player component serve both. `max_frames` evenly spaced scans (all
    if None), read in one streaming pass.

    Real bags carry no odometry-derived x/y/dist and no ground-truth classes,
    so those keys are simply absent from meta here. That is exactly why the
    player derives its telemetry rows from the payload instead of assuming a
    fixed set: the old page assumed, and threw on frame 0.
    """
    out_frames, meta, pick, t0 = [], [], None, None
    info: dict = {}
    for i, (_kind, t_ns, xyz) in enumerate(read_bag(bag, topic, info=info)):
        if pick is None:
            n = info["n_scans"]
            pick = set(np.linspace(0, n - 1, min(max_frames or n, n)).round().astype(int).tolist())
        if i not in pick:
            continue
        d = classify_scan(runner, xyz)
        if isinstance(d, str):
            print(f"  scan {i}: {d}")
            continue
        t0 = t_ns if t0 is None else t0
        out_frames.append(frame_sink(len(out_frames), d["image"]) if frame_sink
                          else encode_frame(d["image"], upscale=upscale))
        meta.append({"t": (t_ns - t0) / 1e9, "mse": d["mse"],
                     **({"pred_class_counts": d["pred_class_counts"]}
                        if d["pred_class_counts"] is not None else {})})
        if len(meta) % progress_every == 0:
            print(f"  processed {len(meta)}/{len(pick)}")

    if not meta:
        raise SystemExit(f"no usable scans on {topic}")
    return {
        "frames": out_frames, "meta": meta,
        "stats": {"n_frames": len(meta), "n_raw_available": info["n_scans"],
                  "duration_s": meta[-1]["t"] - meta[0]["t"],
                  "means": {"mse": float(np.mean([m["mse"] for m in meta]))}},
    }


def write_html(path: Path, frames_b64, meta, bag: Path, runner: ModelRunner):
    """The standalone scrubbable page (viz/player_template.html, patched for
    real data: no position/distance, predicted -- not true -- classes)."""
    template = (REPO_ROOT / "viz" / "player_template.html").read_text(encoding="utf-8")
    html = patch_template(template, {
        "<title>MBUT Corridor Playback</title>": f"<title>Real Bag: {runner.model_name}</title>",
        "LiDAR playback &middot; scenario 6633e5c2": f"Real bag inference &middot; {runner.model_name}",
        "MBUT corridor &mdash; what the encoder sees": f"{bag.name} on real hardware",
        "600 scans &middot; ~318s &middot; ~121m corridor": f"{len(frames_b64)} scans, evenly spaced",
        "Range &amp; class channels, as fed to the model": "Original vs. reconstructed (real data, no ground truth)",
        "Classes in this frame": "Predicted objects (no ground truth on real data)",
        'Scenario 6633e5c2 &middot; sim2real-lidar &middot; class colors: '
        'env dark, sphere amber, cylinder teal, box2 yellow, box1 coral':
            f"{bag.name} &middot; real Ouster OS1-64 bag &middot; class colors: "
            f"env dark, sphere amber, cylinder teal, box2 yellow, box1 coral",
        '<span class="channel-tag range">range</span>': "",
        '<span class="channel-tag class">class</span>': "",
        '<div class="channel-divider"></div>': "",
        '<div class="telemetry-row"><span class="k">Position</span>'
        '<span class="v mono" id="tPos">0.0, 0.0</span></div>': "",
        '<div class="telemetry-row"><span class="k">Distance</span>'
        '<span class="v mono" id="tDist">0.0m</span></div>':
            '<div class="telemetry-row"><span class="k">Recon MSE (norm.)</span>'
            '<span class="v mono" id="tMse">0.0</span></div>',
        "tPos.textContent = m.x.toFixed(1) + ', ' + m.y.toFixed(1);": "",
        "tDist.textContent = m.dist.toFixed(1) + 'm';":
            "document.getElementById('tMse').textContent = m.mse.toFixed(4);",
        "__FRAMES_JSON__": json.dumps({"frames": frames_b64, "meta": meta}),
    })
    path.write_text(html, encoding="utf-8")
    print(f"  wrote {path} ({len(html) / 1e6:.2f} MB, {len(frames_b64)} frames)")


def main():
    ap = argparse.ArgumentParser(description="Offline inference over a recorded ROS2 bag.")
    ap.add_argument("--bag", type=Path, required=True, help="ROS2 bag directory (contains metadata.yaml)")
    ap.add_argument("--topic", default="/ouster/points")
    ap.add_argument("--odom-topic", default=ODOM_TOPIC)
    ap.add_argument("--model", choices=["pointnet2", "salsanext", "kevin_cnn"], default="pointnet2")
    ap.add_argument("--checkpoint", type=Path, help=f"default for pointnet2: {DEFAULT_CHECKPOINT}")
    ap.add_argument("--max-range", type=float, default=14.425209045410156,
                    help="shared normalization constant (outputs/kevin_cnn/normalization_stats.json)")
    ap.add_argument("--class-threshold", type=float, default=0.97,
                    help="Confidence threshold for the predicted-class display -- 0.97 maximizes "
                         "macro-F1 on the labeled simulated test set. 0 = raw argmax.")
    ap.add_argument("--out-dir", type=Path, help="default: outputs/real_inference/<bag name>")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth scan")
    ap.add_argument("--limit", type=int, help="stop after this many scans (smoke runs)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--html", type=Path, help="also write the standalone scrubbable page here")
    ap.add_argument("--max-frames", type=int, default=200, help="scans in the --html page")
    args = ap.parse_args()

    checkpoint = args.checkpoint or (DEFAULT_CHECKPOINT if args.model == "pointnet2" else None)
    if checkpoint is None:
        ap.error(f"--checkpoint is required for {args.model}")
    runner = ModelRunner(args.model, checkpoint, args.max_range, class_conf_threshold=args.class_threshold)
    print(f"Loaded {args.model} from {checkpoint} on {runner.device}")

    summary = run_bag(args.bag, args.out_dir or REPO_ROOT / "outputs" / "real_inference" / args.bag.name,
                      runner, topic=args.topic, odom_topic=args.odom_topic, stride=args.stride,
                      limit=args.limit, video=not args.no_video, html=args.html, max_frames=args.max_frames)
    if summary["status"] != "ok":
        print(f"FAILED: {summary['skippedFraction']:.2%} of scans skipped "
              f"(limit {MAX_SKIPPED_FRACTION:.0%}) -- see the reasons above")
        sys.exit(1)


if __name__ == "__main__":
    main()
