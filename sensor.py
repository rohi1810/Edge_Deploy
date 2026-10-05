"""The one place the sensor's grid shape -- and layout -- are defined.

Both the simulated LiDAR and the real Ouster OS1-64 produce scans on the
same (64 rings x 1024 azimuth columns) grid -- that match is the whole
reason a model trained in simulation can be fed real scans at all (see
ros2_inference/). This shape was previously restated in ten separate
modules under three different names, which is nine chances for a mismatch
nothing would catch.

Same shape is not same ORDER, and order is part of PointNet++'s input: its
ball query keeps the first k in-range points by array index
(models/pointnet2_utils.py, ball_query), so a scan fed in another order is
grouped into different neighbourhoods than anything seen in training.
Measured conventions (simulated .npz scans; real bag3 /ouster/metadata and
point clouds, 2026-09-27):

    simulator (training): row 0 = lowest beam (-22.5 deg), rows go up;
                          col 0 = azimuth -180 deg, columns counter-clockwise
    Ouster (os_lidar):    row 0 = highest beam (+21.6 deg), rows go down;
                          col 0 = azimuth 0 deg, columns clockwise

Both share their axes (z up, the robot drives toward -x), so only the
array order differs, and ouster_to_sim_order() undoes it.

Deliberately NOT imported by tests/conftest.py: a fixture that builds its
test data from the same constant it validates against can't catch that
constant changing, so the test suite keeps its own independent copy.
"""
from __future__ import annotations

import numpy as np

N_RINGS = 64
N_COLS = 1024

# The range the simulated sensor actually produced: 0.300 m to 119.95 m over
# 300 training scans (its configured limits). A real return outside it is
# something the model never saw -- including the (0,0,0) some Ouster driver
# configurations emit for a no-return, and the robot's own body below 0.3 m.
SIM_RANGE_M = (0.30, 120.0)

# Ouster column c looks along azimuth -c*360/N_COLS, simulator column c'
# along -180 + c'*360/N_COLS: the same direction when c = N_COLS/2 - c'.
_OUSTER_TO_SIM_COLS = (N_COLS // 2 - np.arange(N_COLS)) % N_COLS


def ouster_to_sim_order(grid: np.ndarray) -> np.ndarray:
    """(N_RINGS, N_COLS, ...) grid in Ouster order -> simulator order.

    Its own inverse (the row flip and the column map both are), so it also
    turns a simulator-order grid into Ouster order. Keeps column parity, so
    the model's every-2nd-column decimation keeps the same physical columns."""
    return grid[::-1, _OUSTER_TO_SIM_COLS]


def check_sim_order(xyz_grid: np.ndarray) -> None:
    """Raise ValueError unless xyz_grid (N_RINGS, N_COLS, 3) is in simulator
    order: beam elevation rising with the row, azimuth -90/0/+90 deg at
    columns 256/512/768. Refusing is the point: a driver configured
    differently would otherwise feed the model a silently misordered scan."""
    with np.errstate(invalid="ignore"):
        elev = np.degrees(np.arctan2(xyz_grid[..., 2], np.hypot(xyz_grid[..., 0], xyz_grid[..., 1])))
        azim = np.degrees(np.arctan2(xyz_grid[..., 1], xyz_grid[..., 0]))

    rows = [np.median(r[np.isfinite(r)]) for r in elev if np.isfinite(r).sum() >= 16]
    if len(rows) < N_RINGS // 2 or not np.all(np.diff(rows) > 0):
        raise ValueError("scan is not in simulator row order (beam elevation must rise with the "
                         "row index) -- see sensor.py for the conventions")

    for col, want in ((N_COLS // 4, -90.0), (N_COLS // 2, 0.0), (3 * N_COLS // 4, 90.0)):
        a = azim[:, col - 4:col + 5]
        a = a[np.isfinite(a)]
        off = (np.median(a) - want + 180.0) % 360.0 - 180.0 if a.size else np.inf
        if abs(off) > 10.0:
            raise ValueError(f"scan is not in simulator column order (column {col} should look along "
                             f"{want:+.0f} deg, found an offset of {off:+.1f} deg) -- see sensor.py")
