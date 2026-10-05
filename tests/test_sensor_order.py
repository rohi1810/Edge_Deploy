"""sensor.py's Ouster <-> simulator grid-order conventions.

The synthetic scans are built from the conventions as MEASURED (sensor.py's
docstring), with this suite's own grid constants, so a change to the mapping
that breaks them fails here. The mapping was also checked against five scans
from each real bag on 2026-09-28."""
from __future__ import annotations

import numpy as np
import pytest

from sensor import _OUSTER_TO_SIM_COLS, check_sim_order, ouster_to_sim_order

N_RINGS, N_COLS = 64, 1024


def _scan(elev_deg: np.ndarray, az_deg: np.ndarray, r: float = 5.0) -> np.ndarray:
    el, az = np.radians(elev_deg)[:, None], np.radians(az_deg)[None, :]
    z = np.broadcast_to(r * np.sin(el), (len(elev_deg), len(az_deg)))
    return np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), z], axis=-1)


def ouster_scan() -> np.ndarray:
    """Row 0 = top beam; column c looks along azimuth -c * 360/N_COLS."""
    return _scan(np.linspace(21.6, -21.2, N_RINGS), -np.arange(N_COLS) * 360.0 / N_COLS)


def sim_scan() -> np.ndarray:
    """Row 0 = bottom beam; column c looks along azimuth -180 + c * 360/N_COLS."""
    return _scan(np.linspace(-22.5, 22.5, N_RINGS), -180.0 + np.arange(N_COLS) * 360.0 / N_COLS)


def _azimuth(grid):
    return np.degrees(np.arctan2(grid[..., 1], grid[..., 0]))


def test_reorder_is_its_own_inverse():
    g = np.random.default_rng(0).standard_normal((N_RINGS, N_COLS, 3))
    assert np.array_equal(ouster_to_sim_order(ouster_to_sim_order(g)), g)


def test_reorder_keeps_column_parity():
    # The model keeps every 2nd column; parity kept = the same physical rays.
    assert np.array_equal(_OUSTER_TO_SIM_COLS % 2, np.arange(N_COLS) % 2)


def test_reordered_ouster_scan_looks_along_the_simulator_directions():
    got = ouster_to_sim_order(ouster_scan())
    want = sim_scan()
    diff = (_azimuth(got)[0] - _azimuth(want)[0] + 180.0) % 360.0 - 180.0
    assert np.abs(diff).max() < 1e-6
    elev = np.degrees(np.arcsin(got[:, 0, 2] / 5.0))
    assert np.all(np.diff(elev) > 0), "rows must go bottom -> top after the reorder"


def test_check_accepts_simulator_order():
    check_sim_order(sim_scan())
    check_sim_order(ouster_to_sim_order(ouster_scan()))


@pytest.mark.parametrize("bad", [
    ouster_scan(),                    # never reordered
    sim_scan()[::-1],                 # upside down
    sim_scan()[:, ::-1],              # mirrored
])
def test_check_refuses_anything_else(bad):
    with pytest.raises(ValueError):
        check_sim_order(bad)


def test_check_tolerates_missing_returns():
    g = sim_scan().copy()
    g[:, ::3] = np.nan  # a third of every row gone
    g[60:] = np.nan     # and the top rows entirely
    check_sim_order(g)
