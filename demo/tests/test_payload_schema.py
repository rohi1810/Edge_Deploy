"""The demo's payload contract, as tests.

Standard library plus pytest only -- no torch, no numpy, no Pillow -- so this
runs in CI and on any clone, including one with none of the project's data.

Why this exists rather than trusting `build_demo.py --check`: --check is run by
whoever is building. These run by themselves. The five real_bags/*.html players
shipped visibly broken for weeks because nothing asserted the one property that
would have caught it -- the page named telemetry fields (`m.x`,
`m.class_counts`) that the generator never emitted, so render(0) threw before
play() was ever called and autoplay, telemetry and the legend were all silently
dead. test_declared_telemetry_fields_exist_in_every_frame is that bug, encoded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_DIR))

from schema import _load_registered, check_all, check_player_payload  # noqa: E402

PLAYER_FILES = sorted((DEMO_DIR / "payloads" / "players").glob("*.js"))


def player_ids():
    return [p.stem for p in PLAYER_FILES] or ["<none exported>"]


@pytest.fixture(params=PLAYER_FILES, ids=player_ids())
def player(request):
    if not PLAYER_FILES:
        pytest.skip("no player payloads exported yet")
    return request.param


# --------------------------------------------------------------- the contract

def test_declared_telemetry_fields_exist_in_every_frame(player):
    """A payload may only declare a field it can supply for EVERY frame.

    Not "most frames": the player renders frame 0 before anything else, and one
    missing key there takes the whole section down.
    """
    p = _load_registered(player)
    meta = p["meta"]
    for field in p.get("telemetryFields", []):
        keys = field.get("from", [field["key"]])
        for i, m in enumerate(meta):
            missing = [k for k in keys if k not in m]
            assert not missing, (
                f"{player.name}: telemetry field {field['key']!r} reads {keys}, "
                f"but meta[{i}] is missing {missing}")


def test_frame_count_matches_meta(player):
    p = _load_registered(player)
    assert p["frames"]["count"] == len(p["meta"])


def test_referenced_frame_files_exist(player):
    p = _load_registered(player)
    frames = p["frames"]
    if frames.get("mode") != "files":
        pytest.skip("inline frames carry their own data")
    base = DEMO_DIR / "payloads" / frames["base"]
    pad, ext = frames.get("pad", 4), frames.get("format", "png")
    n = frames["count"]
    # First, last and one in the middle. Checking all 160 adds runtime without
    # adding information -- they are written by one loop that cannot skip one.
    for i in {0, n // 2, n - 1}:
        f = base / f"{i:0{pad}d}.{ext}"
        assert f.exists(), f"{player.name}: missing frame file {f}"


def test_timestamps_are_non_decreasing(player):
    """The scheduler computes the next delay as t[i+1] - t[i]. A negative one
    stalls playback rather than erroring, which is far harder to notice."""
    ts = [m["t"] for m in _load_registered(player)["meta"]]
    bad = [i for i in range(1, len(ts)) if ts[i] < ts[i - 1]]
    assert not bad, f"{player.name}: meta[{bad[:3]}].t goes backwards"


def test_legend_source_present_in_every_frame(player):
    p = _load_registered(player)
    legend = p.get("legend")
    if not legend or not legend.get("source"):
        pytest.skip("payload declares no legend")
    src = legend["source"]
    for i, m in enumerate(p["meta"]):
        assert src in m, f"{player.name}: legend source {src!r} missing from meta[{i}]"


def test_real_bag_payloads_do_not_claim_ground_truth(player):
    """Real LiDAR has no labels. A bag payload whose legend calls its classes
    ground truth would be a factual error on the page, not a styling choice."""
    p = _load_registered(player)
    if p.get("kind") != "bag":
        pytest.skip("simulated payload -- ground truth genuinely exists")
    caption = ((p.get("legend") or {}).get("caption") or "").lower()
    assert "ground truth" not in caption or "no ground truth" in caption, (
        f"{player.name}: legend caption {caption!r} claims ground truth on real data")


# ------------------------------------------------------------------ the whole

def test_check_all_reports_no_problems():
    """Same assertions build_demo.py --check runs, so CI cannot pass while a
    manifest section points at a payload or figure that is not there."""
    problems = check_all(DEMO_DIR)
    assert not problems, "\n".join(problems)


def test_manifest_classes_match_the_rendered_palette():
    """The class colours in the manifest are what the table swatches, player
    legends and 3D viewer all read. The frames themselves were coloured by
    viz/playback_common.CLASS_COLORS at render time, so if these drift the page
    labels a colour the pixels do not use."""
    manifest = _load_registered(DEMO_DIR / "payloads" / "manifest.js")
    sys.path.insert(0, str(DEMO_DIR.parent / "viz"))
    try:
        from playback_common import CLASS_COLORS
    except ImportError:  # pragma: no cover - viz/ needs numpy
        pytest.skip("viz/playback_common.py not importable here")

    for entry in manifest["classes"]:
        cid = entry["id"]
        if cid not in CLASS_COLORS:
            continue
        expected = "#%02x%02x%02x" % tuple(CLASS_COLORS[cid])
        assert entry["color"].lower() == expected, (
            f"class {cid} ({entry['name']}): manifest says {entry['color']}, "
            f"frames were rendered with {expected}")


def test_every_section_type_has_a_renderer():
    """A manifest type with no renderer degrades to a visible 'No renderer for
    section type' note -- correct behaviour, but not something to discover on
    the projector."""
    manifest = _load_registered(DEMO_DIR / "payloads" / "manifest.js")
    js = (DEMO_DIR / "assets" / "js" / "sections.js").read_text(encoding="utf-8")
    registry = js.split("var RENDERERS = {", 1)[1].split("};", 1)[0]
    declared = {k.strip().split(":")[0].strip()
                for line in registry.splitlines() for k in line.split(",") if ":" in k}

    def types(sections):
        for s in sections:
            yield s["type"]
            yield from types(s.get("tabs", []))

    for t in types(manifest["sections"]):
        assert t in declared, f"manifest uses section type {t!r}, not in SECTION_RENDERERS"


def test_no_remote_subresources():
    """The demo must render with the network unplugged."""
    from schema import check_no_remote_subresources
    problems = check_no_remote_subresources(DEMO_DIR)
    assert not problems, "\n".join(problems)


def test_payloads_are_valid_json_wrapped_in_register():
    """Every payload must survive the wrapper being peeled off. A stray '*/'
    inside a block comment silently truncated one of these once."""
    for path in [DEMO_DIR / "payloads" / "manifest.js",
                 DEMO_DIR / "payloads" / "metrics.js", *PLAYER_FILES]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert text.lstrip().startswith("/*") or text.lstrip().startswith("DEMO.register"), \
            f"{path.name}: does not start with a comment or DEMO.register"
        assert "DEMO.register(" in text, f"{path.name}: never calls DEMO.register"
        try:
            json.dumps(_load_registered(path))
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"{path.name}: payload is not valid JSON ({type(e).__name__}: {e})")
