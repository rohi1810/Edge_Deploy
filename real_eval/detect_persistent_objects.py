"""Find where PointNet++ persistently sees objects on a real recording.

The point of training in simulation is to get a model that works on real data,
so the model's own predictions are the right instrument for finding objects in
a real bag. What they are NOT is ground truth for scoring that same model --
that would be circular. This script therefore does one thing only: it separates
predictions the model makes CONSISTENTLY, from many viewpoints, from the
speckle it sprays across a single frame.

That separation is itself a sim-to-real measurement. A real object standing in
the corridor is seen from dozens of poses as the robot approaches and passes
it, and every one of those observations lands on the same patch of ground. A
false positive on a floor tile does not survive the robot moving two metres.

Near field only, deliberately: beyond ~8 m the bearing error from odometry yaw
drift smears a cluster across more than a voxel, so a real object would stop
accumulating agreement and look like speckle.

Emits candidate object locations in the odometry frame, each with the number of
DISTINCT scans that agreed on it, plus the fraction of all object points that
join no persistent cluster -- the false-positive rate no simulated metric shows.

    python real_eval/detect_persistent_objects.py --bag real_bags/bag3_20260910_152806
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ros2_inference"))

VOXEL = 0.30        # m -- object footprints are 0.2-0.65 m
MIN_SCANS = 8       # distinct scans that must agree before a voxel counts
MAX_RANGE = 8.0     # m -- near field, so odometry drift cannot smear a cluster
NAMES = {4: "sphere", 5: "cylinder", 6: "box2", 7: "box1"}


def load_odom(bag: Path, topic: str) -> np.ndarray:
    """-> (t_ns, x, y, yaw, cumulative_distance) per odometry message."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    rows = []
    with AnyReader([bag], default_typestore=ts) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            available = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"odometry topic {topic!r} not in bag. Available:\n  "
                             + "\n  ".join(available))
        for conn, t, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            p, q = m.pose.pose.position, m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y * q.y + q.z * q.z))
            rows.append((t, p.x, p.y, yaw))
    a = np.array(rows, dtype=np.float64)
    dist = np.concatenate([[0.0],
                           np.cumsum(np.linalg.norm(np.diff(a[:, 1:3], axis=0), axis=1))])
    return np.column_stack([a, dist])


def merge_voxels(persistent: dict, vox_class: dict) -> list[dict]:
    """8-connected merge of agreeing voxels into object candidates."""
    clusters, unvisited = [], set(persistent)
    while unvisited:
        stack, group = [unvisited.pop()], []
        while stack:
            cx, cy = stack.pop()
            group.append((cx, cy))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cx + dx, cy + dy)
                    if nb in unvisited:
                        unvisited.remove(nb)
                        stack.append(nb)
        scans = set().union(*(set(persistent[g]) for g in group))
        cls = [c for g in group for c in vox_class[g]]
        xs = [g[0] for g in group]
        ys = [g[1] for g in group]
        clusters.append({
            "x": float(np.mean(xs) * VOXEL + VOXEL / 2),
            "y": float(np.mean(ys) * VOXEL + VOXEL / 2),
            "voxels": len(group),
            "points": int(sum(sum(persistent[g].values()) for g in group)),
            "distinctScans": len(scans),
            "extentM": float(max(max(xs) - min(xs) + 1, max(ys) - min(ys) + 1) * VOXEL),
            "classHistogram": {str(c): cls.count(c) for c in sorted(set(cls))},
        })
    clusters.sort(key=lambda c: -c["distinctScans"])
    return clusters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--topic", default="/ouster/points")
    ap.add_argument("--odom-topic", default="/j100_0847/platform/odom/filtered")
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO / "outputs" / "pointnet2_checkpoints"
                    / "epoch39_multitask_model.pt")
    ap.add_argument("--threshold", type=float, default=0.97)
    # The shared normalization constant the encoders were trained under
    # (outputs/kevin_cnn/normalization_stats.json). Only affects the
    # reconstruction branch, which this script does not use, but
    # ModelRunner requires it.
    ap.add_argument("--max-range", type=float, default=14.425209045410156)
    ap.add_argument("--stride", type=int, default=4, help="use every Nth scan")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    from _shared_inference import ModelRunner
    from run_inference_on_bag import iter_pointcloud2

    odom = load_odom(args.bag, args.odom_topic)
    print(f"odometry: {len(odom)} msgs, {odom[-1, 4]:.1f} m of path")
    runner = ModelRunner("pointnet2", args.checkpoint, args.max_range,
                         class_conf_threshold=args.threshold)

    chunks: list = []
    n_obj_pts = n_scans = 0
    per_scan = []

    for i, (t, _h, _w, xyz_grid) in enumerate(iter_pointcloud2(args.bag, args.topic)):
        if i % args.stride:
            continue
        j = min(max(int(np.searchsorted(odom[:, 0], t)), 0), len(odom) - 1)
        ox, oy, yaw, dist = odom[j, 1], odom[j, 2], odom[j, 3], odom[j, 4]

        # Same stride-2 decimation the model was trained on (dataset.COL_STRIDE).
        xyz = xyz_grid[:, ::2].reshape(-1, 3)
        pts = xyz[np.isfinite(xyz).all(axis=-1)]
        if len(pts) < 1000:
            continue
        with torch.no_grad():
            out = runner.model(torch.from_numpy(pts).float().unsqueeze(0).to(runner.device))
            conf, pred = torch.softmax(out["class_logits"][0], dim=-1).max(dim=-1)
            pred = torch.where(conf >= args.threshold, pred,
                               torch.zeros_like(pred)).cpu().numpy()
        n_scans += 1

        rng = np.linalg.norm(pts[:, :2], axis=1)
        obj = (pred >= 4) & (rng < MAX_RANGE) & (rng > 0.4)
        n_obj_pts += int(obj.sum())
        per_scan.append((dist, int(obj.sum())))
        if not obj.any():
            continue

        p = pts[obj]
        c, s = math.cos(yaw), math.sin(yaw)
        wx = ox + c * p[:, 0] - s * p[:, 1]
        wy = oy + s * p[:, 0] + c * p[:, 1]
        # Accumulate as arrays, not as per-point entries in Python dicts.
        # The dict-of-dicts version cost ~36 GB and filled the disk before it
        # could finish: one Python int per predicted object point, across
        # millions of points, is not a viable representation.
        chunks.append(np.column_stack([
            np.floor(wx / VOXEL).astype(np.int32),
            np.floor(wy / VOXEL).astype(np.int32),
            pred[obj].astype(np.int32),
            np.full(obj.sum(), i, dtype=np.int32),
        ]))

        if n_scans % 100 == 0:
            rows = sum(len(c) for c in chunks)
            print(f"  {n_scans} scans, {rows:,} object points accumulated", flush=True)

    if not chunks:
        print("no object predictions at all")
        return 0
    a = np.concatenate(chunks)
    del chunks
    print(f"\naggregating {len(a):,} object points into {VOXEL:.2f} m voxels...")

    # Unique voxel per point, then unique (voxel, scan) pairs -> how many
    # DISTINCT scans agree on each voxel. Both are numpy-side, no Python loop.
    vox_key, vox_idx = np.unique(a[:, :2], axis=0, return_inverse=True)
    pair = np.unique(np.column_stack([vox_idx, a[:, 3]]), axis=0)
    scans_per_vox = np.bincount(pair[:, 0], minlength=len(vox_key))
    pts_per_vox = np.bincount(vox_idx, minlength=len(vox_key))

    keep = scans_per_vox >= MIN_SCANS
    kept = int(pts_per_vox[keep].sum())
    print(f"voxels occupied {len(vox_key):,}, persistent {int(keep.sum()):,}")

    # Drop everything that failed the agreement test BEFORE touching Python
    # objects -- the surviving points are a small fraction of the total.
    vox_idx = np.asarray(vox_idx).ravel()
    sub = a[keep[vox_idx]]
    persistent: dict = {}
    vox_class: dict = defaultdict(list)
    for vx, vy, cl, scan in sub.tolist():
        k = (vx, vy)
        d = persistent.setdefault(k, {})
        d[scan] = d.get(scan, 0) + 1
        vox_class[k].append(cl)
    clusters = merge_voxels(persistent, vox_class)

    print(f"\nscans processed      : {n_scans}")
    print(f"object points total  : {n_obj_pts:,}")
    print(f"in persistent voxels : {kept:,} ({kept / max(n_obj_pts, 1) * 100:.1f}%)")
    print(f"transient (speckle)  : {n_obj_pts - kept:,} "
          f"({(n_obj_pts - kept) / max(n_obj_pts, 1) * 100:.1f}%)")
    print(f"\ncandidates (>= {MIN_SCANS} distinct scans agreeing on the same 30 cm voxel):")
    print(f"{'#':>3} {'x':>8} {'y':>8} {'scans':>6} {'pts':>7} {'size m':>7}  classes")
    for n, cl in enumerate(clusters[:25], 1):
        hist = " ".join(f"{NAMES.get(int(k), k)}={v}"
                        for k, v in cl["classHistogram"].items())
        print(f"{n:>3} {cl['x']:>8.1f} {cl['y']:>8.1f} {cl['distinctScans']:>6} "
              f"{cl['points']:>7} {cl['extentM']:>7.2f}  {hist}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "bag": str(args.bag), "threshold": args.threshold,
            "scans": n_scans, "objectPoints": n_obj_pts, "persistentPoints": kept,
            "voxelM": VOXEL, "minScans": MIN_SCANS, "maxRangeM": MAX_RANGE,
            "clusters": clusters,
            "perScan": [{"dist": float(d), "objPts": int(n)} for d, n in per_scan],
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
