"""The real-data PointNet++ input path (ros2_inference/_shared_inference.py)
and the property it leans on: classification doesn't care where the floor is.

CPU only and synthetic, like the rest of the suite -- no bag, no checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ros2_inference"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared_inference import ModelRunner, SkipScan, real_scan_to_points  # noqa: E402
from dataset import N_POINTS  # noqa: E402
from models.pointnet2 import PointNet2MultiTask  # noqa: E402
from test_sensor_order import ouster_scan  # noqa: E402

N_RINGS, N_COLS = 64, 1024


def _room_scan(seed: int = 0) -> np.ndarray:
    """An Ouster-order scan (see test_sensor_order) with realistic ranges and
    ~8% no-returns."""
    rng = np.random.default_rng(seed)
    g = ouster_scan() * rng.uniform(0.3, 2.0, (N_RINGS, N_COLS, 1))  # 1.5-10 m
    g[rng.random((N_RINGS, N_COLS)) < 0.08] = np.nan
    return g.astype(np.float32)


def test_input_is_exactly_the_training_point_count():
    pts, idx = real_scan_to_points(_room_scan())
    assert pts.shape == (N_POINTS, 3) and pts.dtype == np.float32
    assert len(idx) < N_POINTS  # the rest is padding


def test_each_real_point_comes_from_its_cell():
    g = _room_scan()
    pts, idx = real_scan_to_points(g)
    assert np.array_equal(pts[:len(idx)], g.reshape(-1, 3)[idx])
    assert len(np.unique(idx)) == len(idx)
    assert np.all(idx % N_COLS % 2 == 0), "only even Ouster columns are classified"


def test_returns_outside_the_simulated_range_are_dropped():
    g = _room_scan()
    g[5, 10] = 0.0            # (0,0,0) no-return convention
    g[6, 20] = [150.0, 0, 0]  # beyond the simulator's 120 m
    _, idx = real_scan_to_points(g)
    assert 5 * N_COLS + 10 not in idx and 6 * N_COLS + 20 not in idx


def test_deterministic():
    a, b = real_scan_to_points(_room_scan()), real_scan_to_points(_room_scan())
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_empty_scan_is_skipped_not_crashed():
    with pytest.raises(SkipScan):
        real_scan_to_points(np.full((N_RINGS, N_COLS, 3), np.nan, dtype=np.float32))


def test_class_logits_ignore_a_height_offset():
    # The real sensor sits ~9 cm higher than the simulated one. The encoder
    # sees only neighbourhood-relative coordinates and the decoder only
    # distances, so classification must not care; this pins that down.
    # (Not bit-equality: cdist's matmul formula rounds differently once the
    # coordinates move -- measured: median logit change 3e-6.)
    torch.manual_seed(0)
    model = PointNet2MultiTask().eval()
    xyz = torch.randn(1, 4096, 3)
    with torch.no_grad():
        a = model(xyz)["class_logits"]
        b = model(xyz + torch.tensor([0.0, 0.0, 0.1]))["class_logits"]
    assert (a.argmax(-1) == b.argmax(-1)).float().mean() > 0.999
    assert (a - b).abs().median() < 1e-4


def test_runner_counts_each_classified_point_once(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ckpt = tmp_path / "random.pt"
    torch.manual_seed(0)
    torch.save(PointNet2MultiTask().state_dict(), ckpt)
    runner = ModelRunner("pointnet2", ckpt, 14.425, class_conf_threshold=0.0)

    d = runner.run_detailed(_room_scan())
    assert sum(d["pred_class_counts"].values()) == len(d["pred"]) == len(d["points"]) == len(d["conf"])

    with pytest.raises(ValueError):  # a scan that isn't in Ouster order is refused
        ModelRunner("pointnet2", ckpt, 14.425).run_detailed(
            np.ascontiguousarray(_room_scan()[::-1]))
