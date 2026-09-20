"""Repack the demo as a single shareable page for an Artifact / any host with a
file-count limit.

Why this exists: the folder build stores every playback frame as its own PNG --
680 of them -- because that streams, caches and keeps memory bounded when the
demo is opened as a local folder. Artifact publishing caps a version at 255
files, so the same demo has to be reshaped, not shrunk: each player's frames are
base64-inlined into that player's payload, turning 160-200 files into one.

Measured on the current build: 26.0 MB of PNG becomes 34.7 MB of base64 (the
usual ~33% penalty), total ~50 MB against a 64 MB limit, largest single payload
10.4 MB against a 16 MB per-file limit, and the file count drops 741 -> ~61.

Nothing about the folder build changes. This writes a separate tree.

    python demo/bundle_for_artifact.py [--out DIR]
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO))
from schema import _load_registered  # noqa: E402

# Source-only files that have no business in a published page.
EXCLUDE_NAMES = {
    "build_demo.py", "bundle_for_artifact.py", "schema.py", "sources.json",
    "metrics_template.json", "Dockerfile", ".dockerignore", "nginx-demo.conf",
    "VENDOR.md", "README.md",
}
EXCLUDE_DIRS = {"tests", "__pycache__"}

ARTIFACT_MAX_FILES = 255
ARTIFACT_MAX_TEXT_MB = 16
ARTIFACT_MAX_VERSION_MB = 64

# The hero's title-page block -- author, degree, university, department,
# supervisors, date. Right for a defense handout or an examiner's copy; wrong
# for a link sent to a supervisor, who does not need to be told whose thesis
# this is or who is supervising it. Stripped by default, kept with
# --keep-title-page. `site.robot` is NOT in here: the platform line is about
# the hardware, not the author.
TITLE_PAGE_KEYS = ("author", "degree", "university", "department",
                   "supervisors", "date")


def strip_title_page(manifest_js: Path) -> list[str]:
    """Drop the title-page fields from a copied manifest. The hero's add()
    helper skips falsy values, so removing the keys removes the rows."""
    payload = _load_registered(manifest_js)
    site = payload.get("site", {})
    removed = [k for k in TITLE_PAGE_KEYS if site.pop(k, None) is not None]
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    manifest_js.write_text(
        f'DEMO.register("manifest", "main", {body});\n', encoding="utf-8")
    return removed


def should_copy(rel: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return False
    if rel.name in EXCLUDE_NAMES or rel.suffix == ".pyc":
        return False
    # Frames are inlined into their player payload instead.
    return "players/frames" not in rel.as_posix()


def inline_player(src_js: Path, out_js: Path, payloads_root: Path) -> tuple[int, int]:
    """Rewrite one player payload with its frames base64-inlined.

    Returns (frames inlined, bytes written).
    """
    payload = _load_registered(src_js)
    f = payload.get("frames") or {}
    if f.get("mode") != "files":
        out_js.write_bytes(src_js.read_bytes())
        return 0, out_js.stat().st_size

    # frames.base is relative to payloads/, which is what the page's
    # ctx.resolve() prefixes -- not relative to the player file itself.
    base = payloads_root / f["base"]
    pad, ext = f.get("pad", 4), f.get("format", "png")
    data = []
    for i in range(f["count"]):
        p = base / f"{i:0{pad}d}.{ext}"
        if not p.exists():
            raise SystemExit(f"{src_js.name}: missing frame {p}")
        data.append(base64.b64encode(p.read_bytes()).decode("ascii"))

    payload["frames"] = {"mode": "inline", "format": ext,
                         "count": f["count"], "data": data}
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    out_js.parent.mkdir(parents=True, exist_ok=True)
    out_js.write_text(
        f'DEMO.register("player", {json.dumps(payload["id"])}, {body});\n',
        encoding="utf-8")
    return len(data), out_js.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory (created/replaced)")
    ap.add_argument("--keep-title-page", action="store_true",
                    help="keep author/degree/university/department/supervisors/date "
                         "in the hero (default: strip them -- see TITLE_PAGE_KEYS)")
    args = ap.parse_args()
    out = args.out.resolve()

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    copied = 0
    for src in DEMO.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(DEMO)
        if not should_copy(rel):
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    if not args.keep_title_page:
        removed = strip_title_page(out / "payloads" / "manifest.js")
        print(f"  title page stripped: {', '.join(removed) or '(nothing to strip)'}")

    players = sorted((DEMO / "payloads" / "players").glob("*.js"))
    total_frames = 0
    for p in players:
        n, size = inline_player(p, out / "payloads" / "players" / p.name,
                                DEMO / "payloads")
        total_frames += n
        print(f"  {p.name:<22} {n:>4} frames inlined -> {size/1024/1024:6.2f} MB")

    files = [f for f in out.rglob("*") if f.is_file()]
    total_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
    biggest = max(files, key=lambda f: f.stat().st_size)
    big_mb = biggest.stat().st_size / 1024 / 1024

    print(f"\n  {copied} files copied, {total_frames} frames inlined")
    print(f"  {len(files)} files, {total_mb:.1f} MB total, largest "
          f"{biggest.relative_to(out)} at {big_mb:.2f} MB")

    ok = True
    for label, value, limit in (
        ("file count", len(files), ARTIFACT_MAX_FILES),
        ("total MB", total_mb, ARTIFACT_MAX_VERSION_MB),
        ("largest file MB", big_mb, ARTIFACT_MAX_TEXT_MB),
    ):
        mark = "ok  " if value <= limit else "OVER"
        print(f"  {mark} {label:<16} {value:>8.1f} / {limit}")
        ok &= value <= limit
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
