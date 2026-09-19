"""SalsaNext frame rendering for simulated-scenario playbacks.

The counterpart to gen_playback_pointnet2.py's render_pointnet2_frame and
gen_playback_kevin.py's render_kevin_frame. It did not exist before: SalsaNext
was only ever run on real bags via ros2_inference/_shared_inference.py, which
renders the 3-row real-data layout with no ground-truth class row (real LiDAR
has no laser_retro to show). A three-candidate comparison where only two
candidates have a playable simulated drive invites exactly the question you
don't want at a defense, so this fills the gap.

Four rows, matching the other two candidates: original range, reconstructed
range, ground-truth class, predicted class. No reprojection is needed -- unlike
PointNet++, SalsaNext predicts on the native (64,1024) grid the input came
from, so its output already lines up pixel for pixel.

    python3 viz/gen_playback_salsanext.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.salsanext import SalsaNextMultiTask  # noqa: E402
from playback_common import (N_COLS, N_RINGS, class_rgb, label_row,  # noqa: E402
                              load_native_grids, orient, range_rgb, run_playback)


def render_salsanext_frame(model, device, npz_path, max_range) -> dict:
    """One scan -> the 4-row comparison image plus its scalars.

    Input construction mirrors salsanext_dataset.RangeImageDataset exactly
    (same channel order, same max_range normalization, same zeroing of
    no-return cells) -- if these drift apart the model is being fed something
    it never trained on, and the playback would quietly understate it.
    """
    orig_range_grid, orig_class_grid, xyz_full = load_native_grids(npz_path)

    valid = np.isfinite(orig_range_grid) & (orig_range_grid > 1e-6)
    range_n = np.where(valid, np.clip(orig_range_grid, 0, max_range) / max_range, 0.0)
    x_n = np.where(valid, np.clip(xyz_full[..., 0], -max_range, max_range) / max_range, 0.0)
    y_n = np.where(valid, np.clip(xyz_full[..., 1], -max_range, max_range) / max_range, 0.0)
    z_n = np.where(valid, np.clip(xyz_full[..., 2], -max_range, max_range) / max_range, 0.0)
    inp = np.stack([range_n, x_n, y_n, z_n], axis=0).astype(np.float32)

    with torch.no_grad():
        out = model(torch.from_numpy(inp).unsqueeze(0).to(device))
    recon_n = out["recon_range"][0].cpu().numpy()
    pred_cls = out["class_logits"][0].argmax(dim=0).cpu().numpy().astype(np.uint8)

    recon_display = np.where(valid, recon_n * max_range, np.nan)
    orig_display = np.where(valid, orig_range_grid, np.nan)

    # Scored only where the scan actually returned something: no-return cells
    # carry no ground truth and must not count either way.
    mse = float(np.mean(((range_n - recon_n) ** 2)[valid])) if valid.any() else 0.0
    acc = float((pred_cls[valid] == orig_class_grid[valid]).mean()) if valid.any() else 0.0

    gap = np.full((3, N_COLS, 3), 10, dtype=np.uint8)
    combined = np.concatenate([
        label_row(range_rgb(orient(orig_display)), "ORIGINAL RANGE"), gap,
        label_row(range_rgb(orient(recon_display)), "RECONSTRUCTED RANGE (native grid)"), gap,
        label_row(class_rgb(orient(orig_class_grid)), "ORIGINAL CLASS"), gap,
        label_row(class_rgb(orient(pred_cls)), "PREDICTED CLASS"),
    ], axis=0)

    u, cnt = np.unique(orig_class_grid, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(u.tolist(), cnt.tolist())}

    return {"image": combined, "mse": mse, "acc": acc, "class_counts": class_counts}


def build_replacements(scenario_id, n_frames, duration_s, distance_m,
                        checkpoint_name, frames_json) -> dict[str, str]:
    short = scenario_id[:8]
    return {
        "<title>MBUT Corridor Playback</title>": "<title>SalsaNext Live Inference</title>",
        "LiDAR playback &middot; scenario 6633e5c2":
            f"SalsaNext live inference &middot; scenario {short}",
        "MBUT corridor &mdash; what the encoder sees":
            "SalsaNext on a real drive &mdash; reconstruction &amp; classification",
        "600 scans &middot; ~318s &middot; ~121m corridor":
            f"{n_frames} scans &middot; ~{duration_s:.0f}s &middot; ~{distance_m:.0f}m corridor "
            f"&middot; {checkpoint_name}",
        "Range &amp; class channels, as fed to the model":
            "Original vs. reconstructed range &amp; class (top to bottom)",
        "Classes in this frame": "Objects in this frame (ground truth)",
        'Scenario 6633e5c2 &middot; sim2real-lidar &middot; class colors: '
        'env dark, sphere amber, cylinder teal, box2 yellow, box1 coral':
            f"Scenario {short} &middot; sim2real-lidar &middot; SalsaNext candidate &middot; "
            f"class colors: env dark, sphere amber, cylinder teal, box2 yellow, box1 coral",
        '<span class="channel-tag range">range</span>': "",
        '<span class="channel-tag class">class</span>': "",
        '<div class="channel-divider"></div>': "",
        '<div class="telemetry-row"><span class="k">Distance</span>'
        '<span class="v mono" id="tDist">0.0m</span></div>':
            '<div class="telemetry-row"><span class="k">Distance</span>'
            '<span class="v mono" id="tDist">0.0m</span></div>\n'
            '        <div class="telemetry-row"><span class="k">Recon MSE (norm.)</span>'
            '<span class="v mono" id="tMse">0.0</span></div>\n'
            '        <div class="telemetry-row"><span class="k">Per-pixel class acc.</span>'
            '<span class="v mono" id="tAcc">0.0%</span></div>',
        "tDist.textContent = m.dist.toFixed(1) + 'm';":
            "tDist.textContent = m.dist.toFixed(1) + 'm';\n"
            "    document.getElementById('tMse').textContent = m.mse.toFixed(4);\n"
            "    document.getElementById('tAcc').textContent = (m.acc * 100).toFixed(1) + '%';",
        "__FRAMES_JSON__": frames_json,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-id", default="122e86f0-8b3f-46f5-8eaa-9f6b1232c7df")
    ap.add_argument("--checkpoint", type=Path,
                     default=REPO_ROOT / "outputs" / "salsanext" / "best_multitask_model.pt")
    ap.add_argument("--recordings-dir", type=Path,
                     default=REPO_ROOT / "scenario_data" / "mbut_76scenarios" / "o_robot_nav")
    ap.add_argument("--sim-dir", type=Path, default=REPO_ROOT / "data" / "sim")
    ap.add_argument("--kevin-outputs-dir", type=Path, default=REPO_ROOT / "outputs" / "kevin_cnn")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "salsanext_playback.html")
    ap.add_argument("--max-frames", type=int, default=160)
    ap.add_argument("--upscale", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    max_range = 20.0
    stats_path = args.kevin_outputs_dir / "normalization_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            max_range = json.load(f)["max_range"]
    print(f"max_range: {max_range:.3f}m")

    model = SalsaNextMultiTask().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"loaded checkpoint: {args.checkpoint}")

    run_playback(
        scenario_id=args.scenario_id, recordings_dir=args.recordings_dir, sim_dir=args.sim_dir,
        out_path=args.out, max_frames=args.max_frames or None, upscale=args.upscale,
        checkpoint_name=args.checkpoint.name,
        render_fn=lambda npz_path: render_salsanext_frame(model, device, npz_path, max_range),
        build_replacements=build_replacements,
    )


if __name__ == "__main__":
    main()
