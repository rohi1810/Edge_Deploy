"""Export the demo's payloads from the project's own artifacts.

    python demo/build_demo.py --check          # validate, needs nothing but stdlib
    python demo/build_demo.py --all            # export everything available
    python demo/build_demo.py --only sim_player:pointnet2_sim
    python demo/build_demo.py --report         # per-file sizes + budget check

Design rules worth keeping:

* **No top-level torch import.** Every heavy import happens inside the exporter
  that needs it, so --check and --report run on a laptop with nothing but the
  standard library plus Pillow.
* **Missing inputs skip, they don't fail.** Checkpoints, scenario_data/ and the
  real bags are all gitignored, and Kevin's model lives in a separate sibling
  checkout. A fresh clone must still be able to build the parts it has and tell
  you plainly what it couldn't.
* **Render functions are imported, never copied** -- viz/gen_playback_*.py own
  the per-candidate frame rendering, and this file only arranges the output.
"""
from __future__ import annotations

import argparse
import base64
import importlib
import json
import shutil
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
PAYLOADS = DEMO_DIR / "payloads"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "viz"))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class Skip(Exception):
    """Raised by an exporter when a required input is absent. Reported at the
    end as a skip, not a failure -- see the module docstring."""


def load_sources() -> dict:
    with open(DEMO_DIR / "sources.json", encoding="utf-8") as f:
        return json.load(f)


def need(path: Path, what: str) -> Path:
    if not path.exists():
        raise Skip(f"{what} not found: {path}")
    return path


def write_payload(rel_path: str, kind: str, payload_id: str, payload: dict) -> Path:
    """Write a payload as a classic script calling DEMO.register().

    Not plain .json on purpose: the demo has to work when opened as a file://
    folder, where fetch() is blocked by the opaque origin. A <script> tag is
    not, so every payload is JS that registers itself.
    """
    out = PAYLOADS / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    out.write_text(
        f'DEMO.register({json.dumps(kind)}, {json.dumps(payload_id)}, {body});\n',
        encoding="utf-8",
    )
    return out


def resolve_renderer(spec: str):
    """'module:function' -> the callable, imported only when actually needed."""
    mod_name, fn_name = spec.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def png_frame_sink(out_dir: Path, upscale: int):
    """Writes each frame as its own PNG and returns its index.

    Individual files rather than base64 inlined in the payload: base64 costs
    ~33% more bytes, has to be parsed as one enormous string before anything
    renders, and keeps every frame resident. Files stream, cache, and let the
    player preload a rolling window.
    """
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    def sink(index: int, combined) -> int:
        img = Image.fromarray(combined)
        if upscale != 1:
            img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
        img.save(out_dir / f"{index:04d}.png", format="PNG", optimize=True)
        return index

    return sink


# --------------------------------------------------------------------------
# exporters
# --------------------------------------------------------------------------

def export_sim_player(cfg: dict) -> str:
    """One simulated-scenario playback, rendered by the candidate's own
    render_*_frame function from viz/."""
    import torch  # noqa: F401  (imported here, never at module level)
    from playback_common import (build_playback_payload, derive_legend,
                                 derive_telemetry_fields)

    recordings = REPO_ROOT / "scenario_data" / "mbut_76scenarios" / "o_robot_nav"
    sim_dir = REPO_ROOT / "data" / "sim"
    ckpt = REPO_ROOT / cfg["checkpoint"]
    need(recordings, "scenario_data recordings"); need(sim_dir, "data/sim splits")
    need(ckpt, f"checkpoint for {cfg['id']}")

    render_fn = build_sim_render_fn(cfg, ckpt)

    frames_dir = PAYLOADS / "players" / "frames" / cfg["id"]
    sink = png_frame_sink(frames_dir, cfg.get("upscale", 1)) if cfg.get("frameMode") == "files" else None

    built = build_playback_payload(
        scenario_id=cfg["scenarioId"], recordings_dir=recordings, sim_dir=sim_dir,
        max_frames=cfg.get("maxFrames"), upscale=cfg.get("upscale", 1),
        render_fn=render_fn, frame_sink=sink,
    )
    meta, stats = built["meta"], built["stats"]

    payload = {
        "schemaVersion": 1,
        "id": cfg["id"], "kind": "sim",
        "title": cfg.get("title", cfg["id"]),
        "subtitle": cfg.get("subtitle", ""),
        "source": {
            "model": cfg["model"], "checkpoint": Path(cfg["checkpoint"]).name,
            "scenarioId": cfg["scenarioId"], "bag": None,
            "nFrames": stats["n_frames"], "nRawAvailable": stats["n_raw_available"],
            "durationS": round(stats["duration_s"], 1),
            "distanceM": round(stats["distance_m"], 1),
            "splitCounts": stats["split_counts"],
        },
        "rows": cfg.get("rows", []),
        "telemetryFields": derive_telemetry_fields(meta),
        "legend": derive_legend(meta, caption=cfg.get("legendCaption")),
        "frames": frames_descriptor(cfg, built["frames"]),
        "meta": meta,
        "stats": stats["means"],
    }
    out = write_payload(f"players/{cfg['id']}.js", "player", cfg["id"], payload)
    return f"{cfg['id']}: {stats['n_frames']} frames -> {out.relative_to(DEMO_DIR)}"


def _purge_models_package():
    """Drop any imported top-level `models` package from sys.modules.

    There are TWO packages named `models` in play: this repo's (pointnet2,
    salsanext) and Kevin's, in his separate denkwelt_deploy checkout. Whichever
    is imported first wins for the rest of the process -- which is exactly how
    exporting kevin_sim then salsanext_sim failed with "No module named
    models.salsanext". _shared_inference.py sidesteps this with importlib
    aliasing; here, where each export is independent anyway, clearing the
    binding between candidates is simpler and order-independent.
    """
    for name in [m for m in sys.modules if m == "models" or m.startswith("models.")]:
        del sys.modules[name]


def build_sim_render_fn(cfg: dict, ckpt: Path):
    """Load the candidate's model and bind it into its render_*_frame."""
    import json as _json
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _purge_models_package()
    if cfg["model"] == "kevin_cnn":
        kevin_deploy = (REPO_ROOT.parent / "masterarbeit_kevinfischer" / "lidar_preprocessing"
                        / "autoencoder_training" / "denkwelt_deploy")
        need(kevin_deploy, "Kevin's denkwelt_deploy checkout")
        sys.path.insert(0, str(kevin_deploy))      # his `models` must win here
    else:
        # ...and must not win anywhere else.
        sys.path[:] = [p for p in sys.path if "denkwelt_deploy" not in p]
        if str(REPO_ROOT) not in sys.path[:1]:
            sys.path.insert(0, str(REPO_ROOT))

    render_raw = resolve_renderer(cfg["renderer"])

    stats_path = REPO_ROOT / "outputs" / "kevin_cnn" / "normalization_stats.json"
    max_range = 20.0
    if stats_path.exists():
        with open(stats_path) as f:
            max_range = _json.load(f)["max_range"]

    if cfg["model"] == "pointnet2":
        from models.pointnet2 import PointNet2MultiTask
        model = PointNet2MultiTask().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
    elif cfg["model"] == "kevin_cnn":
        # Kevin's model lives in a sibling checkout that may simply not be
        # present; skipping is the right behaviour, not an ImportError.
        kevin_deploy = (REPO_ROOT.parent / "masterarbeit_kevinfischer" / "lidar_preprocessing"
                        / "autoencoder_training" / "denkwelt_deploy")
        need(kevin_deploy, "Kevin's denkwelt_deploy checkout")
        from gen_playback_kevin import build_autoencoder
        with open(kevin_deploy / "outputs" / "best_params.json") as f:
            best = _json.load(f)
        model = build_autoencoder(best.get("params", best)).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
    elif cfg["model"] == "salsanext":
        from models.salsanext import SalsaNextMultiTask
        model = SalsaNextMultiTask().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        raise Skip(f"no sim renderer wired for model {cfg['model']!r}")

    model.eval()
    return lambda npz_path: render_raw(model, device, npz_path, max_range)


def frames_descriptor(cfg: dict, frames) -> dict:
    if cfg.get("frameMode") == "files":
        return {"mode": "files", "format": "png",
                "base": f"players/frames/{cfg['id']}", "pad": 4,
                "count": len(frames), "preload": 10}
    return {"mode": "inline", "format": "png", "data": frames, "count": len(frames)}


def export_bag_player(cfg: dict) -> str:
    """One real-bag playback. Needs the bag itself (tens of GB, gitignored), so
    this is the exporter most likely to skip on any machine but the one that
    recorded them."""
    import torch  # noqa: F401
    from playback_common import derive_legend, derive_telemetry_fields

    sys.path.insert(0, str(REPO_ROOT / "ros2_inference"))
    from _shared_inference import ModelRunner
    from run_inference_on_bag import bag_frames_payload

    bag = need(REPO_ROOT / cfg["bag"], f"bag for {cfg['id']}")
    ckpt = need(REPO_ROOT / cfg["checkpoint"], f"checkpoint for {cfg['id']}")

    runner = ModelRunner(cfg["model"], ckpt, cfg.get("maxRange", 14.425209045410156),
                         class_conf_threshold=cfg.get("classThreshold", 0.97))

    frames_dir = PAYLOADS / "players" / "frames" / cfg["id"]
    sink = png_frame_sink(frames_dir, cfg.get("upscale", 1)) if cfg.get("frameMode") == "files" else None

    built = bag_frames_payload(
        bag=bag, topic=cfg.get("topic", "/ouster/points"), runner=runner,
        max_frames=cfg.get("maxFrames"), frame_sink=sink, upscale=cfg.get("upscale", 1),
    )
    meta, stats = built["meta"], built["stats"]

    payload = {
        "schemaVersion": 1,
        "id": cfg["id"], "kind": "real",
        "title": cfg.get("title", cfg["id"]),
        "subtitle": cfg.get("subtitle", ""),
        "source": {
            "model": cfg["model"], "checkpoint": Path(cfg["checkpoint"]).name,
            "scenarioId": None, "bag": Path(cfg["bag"]).name,
            "classConfThreshold": cfg.get("classThreshold", 0.97),
            "nFrames": stats["n_frames"], "nRawAvailable": stats["n_raw_available"],
            "durationS": round(stats["duration_s"], 1),
        },
        "rows": cfg.get("rows", []),
        "telemetryFields": derive_telemetry_fields(meta),
        "legend": derive_legend(meta, caption=cfg.get("legendCaption")),
        "frames": frames_descriptor(cfg, built["frames"]),
        "meta": meta,
        "stats": stats["means"],
    }
    out = write_payload(f"players/{cfg['id']}.js", "player", cfg["id"], payload)
    return f"{cfg['id']}: {stats['n_frames']} frames -> {out.relative_to(DEMO_DIR)}"


def read_ply(path: Path) -> dict:
    """Minimal binary-little-endian PLY reader.

    Both assets are `binary_little_endian 1.0` from CloudCompare and Blender,
    so a header-driven numpy structured read covers them without adding a
    parser dependency. Returns {"vertex": structured array, "face": (F,3) int
    or None}. Raises on ascii or big-endian, which nothing here produces.
    """
    import numpy as np

    with open(path, "rb") as f:
        raw = f.read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("latin1")
    if "binary_little_endian" not in header:
        raise Skip(f"{path.name}: only binary_little_endian PLY is supported")

    np_of = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1",
             "uint8": "u1", "char": "i1", "int8": "i1", "short": "<i2",
             "ushort": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4",
             "uint32": "<u4"}

    elements, current = [], None
    for line in header.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "props": [], "list": None}
            elements.append(current)
        elif parts[0] == "property" and current is not None:
            if parts[1] == "list":
                current["list"] = (np_of[parts[2]], np_of[parts[3]], parts[4])
            else:
                current["props"].append((parts[2], np_of[parts[1]]))

    out, off = {}, end
    for elem in elements:
        if elem["list"] is None:
            dt = np.dtype(elem["props"])
            arr = np.frombuffer(raw, dtype=dt, count=elem["count"], offset=off)
            off += dt.itemsize * elem["count"]
            out[elem["name"]] = arr
        else:
            idx, off = _read_face_list(raw, off, elem)
            out[elem["name"]] = idx
    return out


def _read_face_list(raw, off, elem):
    """Read a variable-length face list and fan-triangulate to (F,3).

    Faces genuinely vary in length here: 2.ply is a Blender export whose very
    first face is a quad, and assuming a uniform record size from it asked for
    24.9 MB out of a 19.06 MB block. Record size depends on each face's own
    count, so finding the offsets is inherently sequential -- but that is one
    cheap pass, after which each distinct face length is gathered in bulk.
    """
    import numpy as np

    cnt_dt, idx_dt, _ = elem["list"]
    cnt_size = np.dtype(cnt_dt).itemsize
    idx_size = np.dtype(idx_dt).itemsize
    n_faces = elem["count"]

    counts = np.empty(n_faces, dtype=np.int64)
    starts = np.empty(n_faces, dtype=np.int64)
    pos = off
    for i in range(n_faces):
        c = int.from_bytes(raw[pos:pos + cnt_size], "little")
        counts[i] = c
        starts[i] = pos + cnt_size
        pos += cnt_size + idx_size * c
    end_off = pos

    all_bytes = np.frombuffer(raw, dtype=np.uint8)
    tris = []
    for c in np.unique(counts):
        c = int(c)
        if c < 3:
            continue  # degenerate; nothing to draw
        sel = np.flatnonzero(counts == c)
        byte_idx = starts[sel][:, None] + np.arange(idx_size * c)[None, :]
        block = np.ascontiguousarray(all_bytes[byte_idx])
        idx = block.reshape(-1).view(idx_dt).reshape(len(sel), c).astype(np.int64)
        # fan-triangulate: (v0, vi, vi+1). Correct for the convex quads a
        # Blender export produces; a concave n-gon would need real
        # triangulation, which nothing here emits.
        for k in range(1, c - 1):
            tris.append(idx[:, [0, k, k + 1]])

    faces = np.concatenate(tris, axis=0) if tris else np.zeros((0, 3), dtype=np.int64)
    return faces, end_off


def _quantize(positions, decimals_unused=None):
    """float32 xyz -> int16 plus a per-axis scale/offset.

    Halves the payload against float32 and, at 16 bits across a ~300 m span,
    quantization error is ~5 mm -- far below anything visible in a point cloud
    rendered at a few pixels per point.
    """
    import numpy as np
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scale = span / 65534.0
    q = np.round((positions - lo) / scale).astype(np.int32) - 32767
    return (np.clip(q, -32767, 32767).astype("<i2"),
            [float(v) for v in scale], [float(v) for v in lo])


def _b64(arr) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def export_tunnel(cfg: dict) -> str:
    """The opening act: the SLAM point cloud of the corridor and the
    reconstructed mesh, both as base64 typed arrays.

    Not loaded with three.js's PLYLoader -- that uses fetch/XHR, which is
    blocked on file://, the delivery mode this demo has to support. The
    conversion happens here instead and the viewer decodes typed arrays.

    The two assets are NOT in a shared coordinate frame (cloud spans
    316x100x18.7 m, mesh 128x68x4.2 m, and the mesh's Y extends ~10 m past the
    cloud's), so the viewer cuts between them rather than morphing -- each is
    framed from its own bounding box. Do not "fix" this by scaling one onto
    the other without checking they are actually the same capture.
    """
    import numpy as np

    pc_path = need((REPO_ROOT / cfg["pointcloud"]).resolve(), "tunnel point cloud")
    mesh_path = need((REPO_ROOT / cfg["mesh"]).resolve(), "tunnel mesh")

    # ---- point cloud -----------------------------------------------------
    pc = read_ply(pc_path)["vertex"]
    pts = np.stack([pc["x"], pc["y"], pc["z"]], axis=-1).astype(np.float32)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]

    intensity = None
    for name in ("scalar_intensity", "intensity", "scalar_Intensity"):
        if name in pc.dtype.names:
            intensity = np.asarray(pc[name], dtype=np.float32)[finite]
            break

    max_points = cfg.get("maxPoints", 400000)
    if len(pts) > max_points:
        sel = np.linspace(0, len(pts) - 1, max_points).astype(np.int64)
        pts = pts[sel]
        if intensity is not None:
            intensity = intensity[sel]

    qp, p_scale, p_offset = _quantize(pts)
    cloud = {"count": int(len(pts)), "positions": _b64(qp),
             "scale": p_scale, "offset": p_offset,
             "bounds": {"min": pts.min(axis=0).tolist(), "max": pts.max(axis=0).tolist()},
             "center": pts.mean(axis=0).tolist()}
    if intensity is not None and np.isfinite(intensity).any():
        lo, hi = np.nanpercentile(intensity, [2, 98])
        norm = np.clip((intensity - lo) / max(hi - lo, 1e-6), 0, 1)
        cloud["intensity"] = _b64((norm * 255).astype(np.uint8))

    # ---- mesh ------------------------------------------------------------
    m = read_ply(mesh_path)
    mv, mf = m["vertex"], m["face"]
    verts = np.stack([mv["x"], mv["y"], mv["z"]], axis=-1).astype(np.float32)
    faces = np.asarray(mf, dtype=np.int32)

    colors = None
    if all(c in mv.dtype.names for c in ("red", "green", "blue")):
        colors = np.stack([mv["red"], mv["green"], mv["blue"]], axis=-1).astype(np.uint8)

    max_faces = cfg.get("maxFaces", 200000)
    if len(faces) > max_faces:
        import fast_simplification
        keep = max_faces / len(faces)
        # simplify() returns a remapped vertex set, so per-vertex colours have
        # to be carried across by nearest original vertex rather than reindexed.
        new_v, new_f = fast_simplification.simplify(verts, faces, 1.0 - keep)
        colors = _resample_colors(verts, colors, new_v) if colors is not None else None
        verts, faces = new_v.astype(np.float32), new_f.astype(np.int32)

    qv, m_scale, m_offset = _quantize(verts)
    mesh = {"vertexCount": int(len(verts)), "faceCount": int(len(faces)),
            "positions": _b64(qv), "scale": m_scale, "offset": m_offset,
            "indices": _b64(faces.astype("<u4").ravel()),
            "bounds": {"min": verts.min(axis=0).tolist(), "max": verts.max(axis=0).tolist()},
            "center": verts.mean(axis=0).tolist()}
    if colors is not None:
        mesh["colors"] = _b64(colors.astype(np.uint8).ravel())

    payload = {
        "schemaVersion": 1, "id": cfg["id"],
        "title": cfg.get("title", "Tunnel reconstruction"),
        "cloud": cloud, "mesh": mesh,
        "source": {
            "pointcloudFile": pc_path.name, "meshFile": mesh_path.name,
            "method": cfg.get("method", "Poisson surface reconstruction (CloudCompare)"),
            "sharedFrame": False,
        },
    }
    out = write_payload(f"tunnel/{cfg['id']}.js", "tunnel", cfg["id"], payload)
    return (f"{cfg['id']}: cloud {cloud['count']:,} pts, mesh {mesh['faceCount']:,} faces "
            f"-> {out.stat().st_size/1e6:.1f} MB")


def _resample_colors(old_v, old_c, new_v):
    """Nearest-original-vertex colour lookup after decimation.

    Decimation produces a new vertex set with no index correspondence to the
    original, so colours can't simply be reindexed. A KD-tree if scipy is
    around, otherwise a chunked brute-force search -- this runs once at build
    time, so the slow path is acceptable and avoids a hard dependency.
    """
    import numpy as np
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(old_v).query(new_v, k=1)
        return old_c[idx]
    except ImportError:
        out = np.empty((len(new_v), old_c.shape[1]), dtype=old_c.dtype)
        for start in range(0, len(new_v), 4096):
            chunk = new_v[start:start + 4096]
            d = ((chunk[:, None, :] - old_v[None, :, :]) ** 2).sum(-1)
            out[start:start + 4096] = old_c[d.argmin(axis=1)]
        return out


def export_pointcloud(cfg: dict) -> str:
    """Per-scan 3D point clouds: the real scan coloured by class, plus the
    latent-only reconstruction, for four hand-picked samples (one per object
    class, each the test-split scan containing the most points of it).

    Reads outputs/pointnet2/viz_data.json, which nothing else in the repo
    referenced. Reshapes it from nested [x,y,z] triples into flat Float32-ready
    arrays -- that is what THREE.BufferAttribute wants, and it avoids
    allocating ~500k short-lived JS arrays on load. 2.2 MB becomes ~0.5 MB.
    """
    import numpy as np

    src = need(REPO_ROOT / "outputs" / "pointnet2" / "viz_data.json",
               "outputs/pointnet2/viz_data.json")
    with open(src) as f:
        raw = json.load(f)

    name_to_id = {"sphere": 4, "cylinder": 5, "box2": 6, "box1": 7}
    decimals = cfg.get("decimals", 3)
    samples = []

    for entry in raw:
        env = np.asarray(entry["env"], dtype=np.float32)
        pos_parts, cls_parts = [env], [np.zeros(len(env), dtype=np.uint8)]
        counts = {0: int(len(env))}
        for name, pts in entry["objects"].items():
            arr = np.asarray(pts, dtype=np.float32)
            if not len(arr):
                continue
            cid = name_to_id[name]
            pos_parts.append(arr)
            cls_parts.append(np.full(len(arr), cid, dtype=np.uint8))
            counts[cid] = int(len(arr))

        pos = np.concatenate(pos_parts, axis=0)
        cls = np.concatenate(cls_parts, axis=0)
        recon = np.asarray(entry["recon"], dtype=np.float32)
        allpts = np.concatenate([pos, recon], axis=0)

        samples.append({
            "sampleId": entry["sample_id"],
            "highlightName": entry["highlight"],
            "highlightClass": name_to_id[entry["highlight"]],
            "center": [round(float(v), decimals) for v in allpts.mean(axis=0)],
            "bounds": {"min": [round(float(v), decimals) for v in allpts.min(axis=0)],
                       "max": [round(float(v), decimals) for v in allpts.max(axis=0)]},
            "input": {"count": int(len(pos)),
                      "positions": [round(float(v), decimals) for v in pos.ravel()],
                      "classIds": cls.tolist()},
            "recon": {"count": int(len(recon)),
                      "positions": [round(float(v), decimals) for v in recon.ravel()]},
            "classCounts": counts,
        })

    payload = {
        "schemaVersion": 1, "id": cfg["id"],
        "checkpoint": Path(cfg.get("checkpoint", "")).name,
        "note": ("Reconstruction is latent-only: ReconstructionDecoder(z) with no skip "
                 "connections, so it is a real test of what the 256-d bottleneck retains."),
        "samples": samples,
    }
    out = write_payload(f"pointclouds/{cfg['id']}.js", "pointcloud", cfg["id"], payload)
    return (f"{cfg['id']}: {len(samples)} samples -> "
            f"{out.stat().st_size/1e6:.2f} MB (from {src.stat().st_size/1e6:.1f} MB)")


def export_media(cfg: dict) -> str:
    """Transcode a rendered .mp4 into something a browser can actually play,
    and grab a poster frame.

    This is not optional polish. run_inference_on_bag.py wrote its videos with
    cv2.VideoWriter_fourcc(*"mp4v"), which is MPEG-4 Part 2 -- verified by
    reading the ISO-BMFF sample-entry box on both files. No browser decodes
    that in <video>; it renders a black rectangle with no error. H.264 High/Main
    + yuv420p is the combination that plays everywhere, and +faststart moves
    the index to the front so it begins before the whole file arrives.
    """
    import subprocess
    import imageio_ffmpeg

    src = need(REPO_ROOT / cfg["src"], f"source video for {cfg['id']}")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out_dir = PAYLOADS / "media"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{cfg['id']}.mp4"
    poster = out_dir / f"{cfg['id']}.poster.jpg"

    cmd = [ffmpeg, "-y", "-i", str(src)]
    if cfg.get("maxSeconds"):
        cmd += ["-t", str(cfg["maxSeconds"])]
    cmd += [
        "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
        "-crf", str(cfg.get("crf", 23)), "-preset", "slow",
        "-movflags", "+faststart", "-an", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip().splitlines()[-1] if r.stderr else '?'}")

    subprocess.run([ffmpeg, "-y", "-ss", str(cfg.get("posterAt", 2.0)), "-i", str(out),
                    "-frames:v", "1", "-q:v", "3", str(poster)],
                   capture_output=True, text=True)

    before, after = src.stat().st_size, out.stat().st_size
    return (f"{cfg['id']}: {before/1e6:.1f} MB mp4v -> {after/1e6:.1f} MB h264"
            f" -> {out.relative_to(DEMO_DIR)}")


FONT_FACES = [
    ("IBM Plex Sans", 400, "IBMPlexSans-400.woff2"),
    ("IBM Plex Sans", 600, "IBMPlexSans-600.woff2"),
    ("IBM Plex Mono", 400, "IBMPlexMono-400.woff2"),
    ("IBM Plex Mono", 500, "IBMPlexMono-500.woff2"),
]


def export_fonts(cfg: dict = None) -> str:
    """Generate assets/css/fonts.css with the woff2 files inlined as data: URIs.

    data: rather than url(../fonts/x.woff2) because several Chrome builds
    refuse to load font files over file://, which is precisely the delivery
    mode an examiner will use -- and the failure is silent, just a fallback to
    system-ui. The base64 overhead (~33%) is a few tens of KB here and buys
    typography that cannot quietly disappear.
    """
    fonts_dir = DEMO_DIR / "assets" / "fonts"
    need(fonts_dir, "assets/fonts directory")

    blocks = [
        "/* GENERATED FILE -- do not edit by hand.",
        " * Rebuild with:  python demo/build_demo.py --fonts",
        " *",
        " * IBM Plex, latin subset, inlined as data: URIs so the page keeps its",
        " * typography when opened from a folder with no network. See VENDOR.md.",
        " * Licence: SIL OFL 1.1, shipped alongside in assets/fonts/.",
        " */",
        "",
    ]
    total = 0
    for family, weight, fname in FONT_FACES:
        path = fonts_dir / fname
        if not path.exists():
            raise Skip(f"font file missing: {fname}")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        total += path.stat().st_size
        blocks.append(
            "@font-face {\n"
            f"  font-family: '{family}';\n"
            "  font-style: normal;\n"
            f"  font-weight: {weight};\n"
            "  font-display: swap;\n"
            f"  src: url(data:font/woff2;base64,{data}) format('woff2');\n"
            "}\n"
        )

    out = DEMO_DIR / "assets" / "css" / "fonts.css"
    out.write_text("\n".join(blocks), encoding="utf-8")
    return (f"fonts: {len(FONT_FACES)} faces, {total/1024:.0f} KB raw "
            f"-> {out.stat().st_size/1024:.0f} KB css")


def export_gallery(cfg: dict) -> str:
    copied = 0
    for src_rel in cfg.get("from", []):
        src = REPO_ROOT / src_rel
        if not src.exists():
            continue
        dest = PAYLOADS / "gallery" / Path(src_rel).name
        dest.mkdir(parents=True, exist_ok=True)
        for png in sorted(src.glob("*.png")):
            shutil.copy2(png, dest / png.name)
            copied += 1
    if not copied:
        raise Skip("no gallery PNGs found under outputs/pointnet2/")
    return f"gallery: copied {copied} figures"


# --------------------------------------------------------------------------
# validation / reporting
# --------------------------------------------------------------------------

def cmd_check() -> int:
    from schema import check_all
    problems = check_all(DEMO_DIR)
    for p in problems:
        print("  FAIL " + p)
    if problems:
        print(f"\n{len(problems)} problem(s).")
        return 1
    print("check: OK")
    return 0


def cmd_report() -> int:
    total = 0
    rows = []
    for path in sorted(DEMO_DIR.rglob("*")):
        if path.is_dir() or ".git" in path.parts:
            continue
        size = path.stat().st_size
        total += size
        rows.append((size, path.relative_to(DEMO_DIR)))
    rows.sort(reverse=True)

    print(f"{'size':>12}  file")
    for size, rel in rows[:25]:
        print(f"{size/1e6:>9.2f} MB  {rel}")
    if len(rows) > 25:
        smaller = sum(s for s, _ in rows[25:])
        print(f"{smaller/1e6:>9.2f} MB  ... and {len(rows)-25} smaller files")
    print(f"\n{'total':>9}: {total/1e6:.2f} MB across {len(rows)} files")

    # GitHub refuses any file at/over 100 MB, and Pages serves LFS pointers as
    # text rather than resolving them, so a big file is a hard stop not a nudge.
    bad = [rel for size, rel in rows if size >= 95e6]
    if bad:
        print("\nFAIL: files at/near GitHub's 100 MB hard limit:")
        for rel in bad:
            print(f"  {rel}")
        return 1
    if total >= 200e6:
        print(f"\nFAIL: total {total/1e6:.0f} MB exceeds the 200 MB budget.")
        return 1
    return 0


# --------------------------------------------------------------------------

EXPORTERS = {
    "sim_player": ("simPlayers", export_sim_player),
    "bag_player": ("bagPlayers", export_bag_player),
    "media": ("media", export_media),
    "gallery": ("gallery", export_gallery),
    "tunnel": ("tunnel", export_tunnel),
    "pointcloud": ("pointclouds", export_pointcloud),
}


def run_exports(sources: dict, only: list[str] | None) -> int:
    done, skipped, failed = [], [], []

    def selected(kind: str, ident: str | None) -> bool:
        if not only:
            return True
        for sel in only:
            if sel == kind:
                return True
            if ":" in sel and sel.split(":", 1) == [kind, ident]:
                return True
        return False

    for kind, (src_key, fn) in EXPORTERS.items():
        cfgs = sources.get(src_key)
        if cfgs is None:
            continue
        items = cfgs if isinstance(cfgs, list) else [cfgs]
        for cfg in items:
            ident = cfg.get("id", src_key) if isinstance(cfg, dict) else src_key
            if not selected(kind, ident):
                continue
            try:
                done.append(fn(cfg))
            except Skip as e:
                skipped.append(f"{kind}:{ident} -- {e}")
            except Exception as e:  # noqa: BLE001 -- report and continue
                failed.append(f"{kind}:{ident} -- {type(e).__name__}: {e}")

    print("\n--- export summary ---")
    for d in done:
        print(f"  ok      {d}")
    for s in skipped:
        print(f"  skipped {s}")
    for f in failed:
        print(f"  FAILED  {f}")
    if skipped and not failed:
        print("\nSkips are expected on a machine without the full dataset; "
              "prior payloads (if any) were left untouched.")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run every exporter")
    ap.add_argument("--only", help="comma-separated kind or kind:id selectors")
    ap.add_argument("--check", action="store_true", help="validate payloads, no torch needed")
    ap.add_argument("--report", action="store_true", help="file sizes and budget check")
    ap.add_argument("--fonts", action="store_true", help="regenerate assets/css/fonts.css")
    args = ap.parse_args()

    if args.check:
        return cmd_check()
    if args.report:
        return cmd_report()
    if args.fonts:
        print(export_fonts())
        return 0
    if not (args.all or args.only):
        ap.print_help()
        return 0

    sources = load_sources()
    only = args.only.split(",") if args.only else None
    return run_exports(sources, only)


if __name__ == "__main__":
    raise SystemExit(main())
