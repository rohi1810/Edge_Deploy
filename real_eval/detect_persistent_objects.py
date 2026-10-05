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

    python real_eval/detect_persistent_objects.py outputs/real_inference/bag3_20260910_152806

Reads the run directory ros2_inference/run_inference_on_bag.py wrote
(predictions.npz: object points per scan, odometry, downsampled map scans) --
no model, no second pass over the bag -- and writes objects.json and
objects_map.png next to it: candidate object locations in the odometry frame,
each with the number of DISTINCT scans that agreed on it and when it was seen,
plus the share of object points that join no persistent cluster -- the
false-positive rate no simulated metric shows.

The Ouster's +x axis points BACKWARD on this robot (os_lidar; see sensor.py),
and the bag records no base-to-sensor transform, so the sensor's yaw on the
robot is measured from the data itself (estimate_mount_yaw). Until
2026-09-28 this script assumed +x was forward, which put every static object
at 2 x robot - object, so it could never collect votes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from dataset import CLASS_NAMES  # noqa: E402

VOXEL = 0.30        # m -- object footprints are 0.2-0.65 m
MIN_SCANS = 8       # distinct scans that must agree before a voxel counts
NEAR = (0.4, 8.0)   # m, horizontal -- near field, so odometry drift cannot smear a cluster
MAP_VOXEL = 0.2     # m, for the mount-yaw fit
SCATTER_RADIUS, SCATTER_MIN_NEIGHBOURS = 0.25, 3

# The videos' class colours (viz/playback_common.CLASS_COLORS), so a map and
# a video of the same bag agree.
CLASS_HEX = {4: "#f2a65a", 5: "#4fd1c5", 6: "#f0d65a", 7: "#eb6e5a"}
SURFACE, INK, INK2, MUTED, MAPDOT = "#1a1a19", "#ffffff", "#c3c2b7", "#898781", "#4a4a46"


def poses_at(odom: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
    """Odometry (x, y, yaw) interpolated at t_ns; NaN outside its time span."""
    t = odom[:, 0]
    pose = np.stack([np.interp(t_ns, t, odom[:, 1]), np.interp(t_ns, t, odom[:, 2]),
                     np.interp(t_ns, t, np.unwrap(odom[:, 3]))], axis=-1)
    pose[(t_ns < t[0]) | (t_ns > t[-1])] = np.nan
    return pose


def to_world(xy: np.ndarray, pose: np.ndarray, mount_yaw: float) -> np.ndarray:
    """Sensor-frame xy -> odometry frame. pose: (n, 3) per point or (3,)."""
    a = pose[..., 2] + mount_yaw
    c, s = np.cos(a), np.sin(a)
    return np.stack([pose[..., 0] + c * xy[:, 0] - s * xy[:, 1],
                     pose[..., 1] + s * xy[:, 0] + c * xy[:, 1]], axis=-1)


def estimate_mount_yaw(xy: np.ndarray, pose: np.ndarray, scan: np.ndarray, window: int = 60) -> float:
    """The sensor's yaw on the robot, from the data: the angle at which the
    stored scans, placed by odometry, overlap best -- fewest occupied
    MAP_VOXEL cells, counted within windows of `window` stored scans (about a
    minute) so slow heading drift can't blur the answer. A wrong angle smears
    every door frame, corner and object; a corridor's walls alone would not
    settle it, the rest of the scene does."""
    win = (scan // window).astype(np.int64) << 42

    def cost(theta):
        v = np.floor(to_world(xy, pose, theta) / MAP_VOXEL).astype(np.int64) & 0x1FFFFF
        return len(np.unique(win | v[:, 0] << 21 | v[:, 1]))

    coarse = np.radians(np.arange(0.0, 360.0, 5.0))
    best = coarse[np.argmin([cost(t) for t in coarse])]
    fine = best + np.radians(np.arange(-5.0, 5.01, 0.25))
    return float(fine[np.argmin([cost(t) for t in fine])] % (2 * np.pi))


def merge_voxels(persistent: dict, vox_class: dict, t_s: np.ndarray) -> list[dict]:
    """8-connected merge of agreeing voxels into object candidates."""
    clusters, unvisited = [], set(persistent)
    while unvisited:
        stack, group = [unvisited.pop()], []
        while stack:
            cx, cy = stack.pop()
            group.append((cx, cy))
            nbs = unvisited & {(cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
            unvisited -= nbs
            stack.extend(nbs)
        scans = sorted(set().union(*(persistent[g] for g in group)))
        cls = Counter(c for g in group for c in vox_class[g])
        top, n_top = cls.most_common(1)[0]
        xs, ys = zip(*group)
        clusters.append({
            "x": float((np.mean(xs) + 0.5) * VOXEL),
            "y": float((np.mean(ys) + 0.5) * VOXEL),
            "voxels": len(group),
            "points": sum(cls.values()),
            "distinctScans": len(scans),
            "extentM": float(max(max(xs) - min(xs), max(ys) - min(ys)) + 1) * VOXEL,
            "class": CLASS_NAMES[top],
            "classId": int(top),
            "classPurity": n_top / sum(cls.values()),
            "classHistogram": {CLASS_NAMES[c]: n for c, n in sorted(cls.items())},
            "firstSeenS": float(t_s[scans[0]]),
            "lastSeenS": float(t_s[scans[-1]]),
        })
    clusters.sort(key=lambda c: -c["distinctScans"])
    return clusters


def analyze(run_dir: Path, stride: int = 4, threshold: float = 0.97, plot: bool = True,
            mount_yaw: float | None = None) -> dict:
    """objects.json's content for one inference run directory. mount_yaw
    (radians): reuse a value measured on another run of the same bag."""
    p = np.load(run_dir / "predictions.npz")
    if len(p["odom"]) < 2:
        raise SystemExit(f"{run_dir}: no odometry in predictions.npz -- can't place anything")
    use = np.flatnonzero(p["scan_index"] % stride == 0)
    t_s = (p["t_ns"] - p["t_ns"][0]) / 1e9
    pose = poses_at(p["odom"], p["t_ns"])

    # Mount yaw from the stored map scans (every 10th scan), with odometry.
    m_rows = np.repeat(p["map_row"], np.diff(p["map_offsets"]))
    ok = np.isfinite(pose[m_rows, 0])
    if mount_yaw is None:
        mount_yaw = estimate_mount_yaw(p["map_xyz"][ok, :2], pose[m_rows[ok]], m_rows[ok] // 10)
    yaw_deg = float(np.degrees(mount_yaw))
    if abs(yaw_deg - 180.0) > 10.0:
        raise SystemExit(f"{run_dir}: measured sensor yaw {yaw_deg:.1f} deg, expected ~180 (os_lidar +x "
                         f"points backward on this robot) -- stopping rather than map with it")

    # Object points at the display threshold, from the scans in use.
    off = p["obj_offsets"]
    rows = np.repeat(np.arange(len(off) - 1), np.diff(off))
    keep = np.isin(rows, use) & (p["obj_conf"].astype(np.float32) >= threshold) & np.isfinite(pose[rows, 0])
    xyz, cls, rows = p["obj_xyz"][keep], p["obj_pred"][keep], rows[keep]

    # Scattered: object points with too few object neighbours in their own scan.
    scattered = 0
    for r in np.unique(rows):
        q = xyz[rows == r]
        n_near = cKDTree(q).query_ball_point(q, SCATTER_RADIUS, return_length=True) - 1
        scattered += int((n_near < SCATTER_MIN_NEIGHBOURS).sum())

    h = np.hypot(xyz[:, 0], xyz[:, 1])
    near = (h > NEAR[0]) & (h < NEAR[1])
    world = to_world(xyz[near, :2], pose[rows[near]], mount_yaw)
    keys = np.floor(world / VOXEL).astype(np.int64)
    vox, inv = np.unique(keys, axis=0, return_inverse=True)
    pair = np.unique(np.column_stack([inv, rows[near]]), axis=0)
    persist = (np.bincount(pair[:, 0], minlength=len(vox)) >= MIN_SCANS)[inv]

    persistent, vox_class = defaultdict(set), defaultdict(list)
    for (vx, vy), cl, r in zip(keys[persist].tolist(), cls[near][persist].tolist(), rows[near][persist].tolist()):
        persistent[vx, vy].add(r)
        vox_class[vx, vy].append(cl)
    clusters = merge_voxels(persistent, vox_class, t_s)

    n_scans, n_obj, n_near = len(use), len(xyz), int(near.sum())
    result = {
        "run": str(run_dir), "threshold": threshold, "stride": stride, "scansAnalysed": n_scans,
        "mountYawDeg": yaw_deg, "voxelM": VOXEL, "minScans": MIN_SCANS, "nearFieldM": NEAR,
        "objectPointsPerScan": n_obj / max(n_scans, 1),
        "scatteredShare": scattered / max(n_obj, 1),
        "persistentShare": int(persist.sum()) / max(n_near, 1),
        "clusters": clusters,
    }
    if plot:
        plot_map(p, pose, mount_yaw, clusters, result, run_dir / "objects_map.png")
    return result


def plot_map(p, pose, mount_yaw, clusters, result, path: Path, label_top: int = 20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m_rows = np.repeat(p["map_row"], np.diff(p["map_offsets"]))
    ok = np.isfinite(pose[m_rows, 0])
    walls = to_world(p["map_xyz"][ok, :2], pose[m_rows[ok]], mount_yaw)
    path_xy = pose[np.isfinite(pose[:, 0]), :2]

    span = np.ptp(walls, axis=0) + 4
    w = 9.0
    fig, (ax, side) = plt.subplots(
        1, 2, figsize=(w + 3.6, max(5.0, min(11.0, w * span[1] / span[0]))), dpi=150,
        facecolor=SURFACE, gridspec_kw={"width_ratios": [w, 3.6]})
    ax.set_facecolor(SURFACE)
    side.set_facecolor(SURFACE)
    side.axis("off")
    ax.scatter(walls[:, 0], walls[:, 1], s=0.4, c=MAPDOT, linewidths=0, rasterized=True)
    ax.plot(path_xy[:, 0], path_xy[:, 1], color=MUTED, linewidth=0.8, label="robot path (odometry)")
    for c in CLASS_HEX:
        cs = [k for k in clusters if k["classId"] == c]
        if cs:
            ax.scatter([k["x"] for k in cs], [k["y"] for k in cs],
                       s=[12 + 3 * min(k["distinctScans"], 60) for k in cs], c=CLASS_HEX[c],
                       edgecolors=SURFACE, linewidths=0.8, zorder=3, label=f"{CLASS_NAMES[c]} ({len(cs)})")
    # Numbers on the map, details beside it: clusters crowd together, sentences collide.
    lines = []
    for n, k in enumerate(clusters[:label_top], 1):
        ax.annotate(f"{n}", (k["x"], k["y"]), xytext=(4, 3), textcoords="offset points",
                    fontsize=6.5, color=INK)
        lines.append(f"{n:>2}  {k['class']:<8} {k['distinctScans']:>3} scans  "
                     f"{k['firstSeenS']:>4.0f}-{k['lastSeenS']:<4.0f}s  {k['classPurity']:.0%}")
    side.text(0.0, 1.0, "  #  class    agreeing   seen (s)   same\n"
                        "              scans    into bag   class\n" + "\n".join(lines)
              + (f"\n\n+{len(clusters) - label_top} smaller clusters" if len(clusters) > label_top else ""),
              va="top", ha="left", fontsize=6.8, family="monospace", color=INK2, transform=side.transAxes)
    ax.set_aspect("equal")
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(color="#2c2c2a", linewidth=0.5)
    ax.set_xlabel("odometry x (m)", color=MUTED, fontsize=8)
    ax.set_ylabel("odometry y (m)", color=MUTED, fontsize=8)
    ax.set_title(f"{Path(result['run']).name}: where PointNet++ keeps seeing objects "
                 f"(>= {MIN_SCANS} of {result['scansAnalysed']} scans agree, conf >= {result['threshold']:g})\n"
                 f"{len(clusters)} persistent clusters; {result['persistentShare']:.0%} of near-field object "
                 f"points persistent; {result['scatteredShare']:.0%} of all object points scattered; "
                 f"no ground truth", color=INK, fontsize=8.5, loc="left")
    leg = ax.legend(loc="best", fontsize=7, frameon=False, labelcolor=INK2, markerscale=0.7)
    for h in leg.legend_handles:
        if hasattr(h, "set_sizes"):
            h.set_sizes([30])
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Persistent object detections from a bag inference run.")
    ap.add_argument("run_dir", type=Path, help="outputs/real_inference/<bag> (has predictions.npz)")
    ap.add_argument("--stride", type=int, default=4,
                    help="use every Nth scan of the bag (4 = 2.5 Hz, what MIN_SCANS is sized for)")
    ap.add_argument("--threshold", type=float, default=0.97)
    args = ap.parse_args()

    r = analyze(args.run_dir, args.stride, args.threshold)
    (args.run_dir / "objects.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(f"{args.run_dir.name}: sensor yaw on robot {r['mountYawDeg']:.2f} deg; {r['scansAnalysed']} scans; "
          f"{r['objectPointsPerScan']:.1f} object points/scan; scattered {r['scatteredShare']:.1%}; "
          f"persistent (near field) {r['persistentShare']:.1%}; {len(r['clusters'])} clusters")
    print(f"{'#':>3} {'x':>8} {'y':>8} {'scans':>6} {'pts':>6} {'size m':>7} {'seen s':>13}  class (purity)")
    for n, c in enumerate(r["clusters"][:25], 1):
        print(f"{n:>3} {c['x']:>8.1f} {c['y']:>8.1f} {c['distinctScans']:>6} {c['points']:>6} "
              f"{c['extentM']:>7.2f} {c['firstSeenS']:>6.0f}-{c['lastSeenS']:<6.0f}  "
              f"{c['class']} ({c['classPurity']:.0%})")
    print(f"wrote {args.run_dir / 'objects.json'} and objects_map.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
