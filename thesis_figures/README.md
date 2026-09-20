# Chapter 5 figures — provenance

Regenerate all three with:

```bash
python thesis_figures/make_thesis_figures.py
```

Every number is traced to a file below. Nothing was estimated, interpolated or
invented. Where the data does not exist on this machine, the figure says so on
its own face rather than being filled in.

---

## `fig1_drift_fix_comparison.png`

**Source: the thesis PDF itself — Chapter 5 prose, §5.2.5, §5.3.2, §5.3.4, and its Figure 5.4.**

These are already-aggregated summary statistics copied out of the document, not
re-derived from logs.

> **The `jackal-tunnel-slam` / `jackal-slam-to-mesh` repository is not on this
> machine.** Searched `C:\`, `D:\`, `E:\` for the repo, for
> `odom_covariance_debug.py`, for `tools/diag_*.py`, and for any `*odom*`,
> `*covariance*`, `*drift*`, `*fastlio*` or `*kiss*` file. Nothing. There is no
> per-message CSV to re-run or load.

Because of that, three requested panels **could not be produced** and are absent
rather than faked:

| requested panel | why it is missing |
|---|---|
| per-step jump magnitude vs time (log) | needs per-message position logs |
| before/after X–Y trajectory plots | needs the per-message trajectory |
| jump magnitude vs rotation-rate scatter | needs angular velocity logged per message |

Values plotted, each with the sentence it came from:

| condition | jumps > 1 m | largest jump | extent | median |
|---|---|---|---|---|
| Original | 285 | 12.2 km | 24.0 km | not reported |
| Timestamp fix only | 3,932 | 691.83 m | 25.9 km | not reported |
| Both fixes | 0 | 5.5 mm | 34.62 m | 3.4 mm |

- §5.2.5 — *"285 jumps larger than one metre, the largest reaching 12.2 km, and a
  total trajectory extent of 24.0 km"*
- §5.2.5 — *"3,932 jumps, the largest reaching 691.83 m"*
- §5.3.4 — *"zero jumps larger than one metre, a largest single jump of 5.5 mm, a
  median jump of 3.4 mm, and a trajectory extent of 34.62 m ... down from a
  24.0 km trajectory extent before any fix and 25.9 km with the timestamp fix
  alone over the same window"* (150 s evaluation window)
- Side panel also uses §5.3.4 (*"2.8 Hz to approximately 108 Hz"*), §5.3.2
  (*"−218 parts per million"*) and §5.2.5 (*"1.07 ms instead of the true value of
  approximately 100 ms"*).

**Cross-check against the numbers you supplied: all match the thesis exactly.**

---

## `fig2_reconstructed_pointcloud.png`

**Source: `noslil.ply`** (repo root) — 482,757 vertices, `x, y, z,
scalar_intensity`, binary little-endian, written by CloudCompare v2.14.alpha,
created 2026-09-19. Read directly by `read_ply_xyz_intensity()`; every number in
the figure is measured from the file at run time.

Two things are worth knowing before this goes in the thesis:

1. **The raw bounding box overstates the reconstruction.** Full bbox is
   315.92 × 100.50 × 18.67 m, but 99% of the cloud lies within
   **124.94 × 34.60 × 6.08 m**. 2,591 points (0.54%) sit outside the plotted
   window and stretch the box. The figure reports **both**, and the views are
   cropped to the 0.1–99.9th percentile. Quoting "316 m extent" alone would be
   misleading.
2. **The voxel downsample uses 20 cm, not 5 cm.** Measured on this file: 2 cm,
   5 cm and 10 cm voxels remove essentially nothing (482,757 → 482,750 at 10 cm)
   because the cloud was already decimated to roughly 10 cm spacing before
   export. 20 cm gives the first real reduction, to 143,011 points. Both numbers
   are shown so the reduction is not overstated.

Colour is height above a floor reference taken as the 1st-percentile Z
(−0.13 m); the ramp spans 0–6.08 m (99th percentile of height).

> Note: you previously said `noslil.ply` is a stand-in until a cleaner scan is
> captured. Swap the file and re-run — nothing else needs to change.

Other geometry on this machine, **not** used for this figure:
`MBUTMESH/MBUTMESH.ply` (841,234 verts / 1,680,544 faces — the CloudCompare
Poisson mesh), `2.ply` (Blender export with vertex colours + UVs),
`Mesh[25.ply` (an earlier, smaller mesh). Say the word if you want the mesh
rendered instead of, or beside, the cloud.

---

## `fig3_encoder_comparison.png`

**Sources:**

- `outputs/measured/fair_comparison.json` — written by
  `kevin_pipeline/fair_comparison.py --json-out`, run on the 1,140-sample
  held-out test split. Supplies MSE, SSIM and per-point accuracy for all three.
- `outputs/measured/evaluate_pointnet2.json` — written by
  `evaluate.py --class-conf-threshold 0.97 --json-out`. Supplies macro-F1 and
  the per-class F1 values.

> **No per-epoch training curves are plotted, because none exist.**
> `train_utils.fit()` hands each epoch to a `log_fn` that only `print()`s to
> stdout (`train.py:108`, `train_salsanext.py:103`) — nothing writes history to
> disk. Every `outputs/*/train_metrics.json` holds final scalars only
> (`held_out_val_recon`, `held_out_val_class`, `best_val_loss`, class counts,
> args) with no per-epoch array. No `.log` or `.csv` capture of a training run
> exists anywhere in the repo. So this is the grouped final-metric comparison,
> as you specified for that case.

| candidate | MSE | SSIM | accuracy |
|---|---|---|---|
| PointNet++ | 0.000171 | 0.9716 | 98.96% |
| CNN autoencoder | 0.000664 | 0.9569 | no classifier |
| SalsaNext | 0.001288 | 0.9524 | 98.93% |

**Cross-check against the numbers you supplied: all match exactly.**

Two presentation choices stated on the figure rather than hidden:

- The SSIM axis starts at 0.940. All three land in 0.952–0.972, so a zero-based
  axis renders three visually identical bars.
- The CNN autoencoder shows *"no classifier"*, not a zero bar — it has no
  classification head at all, and a zero would read as "got everything wrong".

---

## `fig4_jetson_realtime.png`

**Source:** `benchmarks/pointnet2_jetson_bench.py`, run on the real Jetson AGX Orin;
values as reported in thesis §5.7.2 (latency and peak memory at five point counts).
`PointNet2Encoder` in isolation, single-scan batches.

Caveat carried onto the figure: these were measured **before** the determinism and
ball-query fixes. Those replace a masked array operation with an equivalent unmasked one
rather than changing the algorithm, so the timings are not expected to move — but that
has not been re-confirmed on the current code.

---

## `fig5_decimation_validation.png`

**Source:** thesis §5.6.4 / Table 5.1 — per-class frame coverage for four decimation
patterns, measured against the full recording. The same numbers appear in `README.md`.
Percentages are the table's frame counts out of 600 recorded frames.

The full-resolution row is the baseline (no decimation), not a competing strategy; the
figure labels it as such so it is not read as a fourth candidate.

---

## `fig6_confidence_threshold.png`

**Source:** `outputs/measured/evaluate_pointnet2_sweep.json`, written by

```bash
python evaluate.py --class-conf-threshold 0.97 \
    --checkpoint outputs/pointnet2_checkpoints/epoch39_multitask_model.pt \
    --json-out outputs/measured/evaluate_pointnet2.json \
    --sweep "0,0.3,0.5,0.7,0.8,0.9,0.95,0.97,0.99,0.995"
```

`--sweep` scores every threshold in **one** forward pass. The confidence value is already
produced by the network; only the comparison against it changes, so sweeping costs one
confusion matrix per threshold rather than one full evaluation pass per threshold.

Measured macro-F1 over the four object classes:

| threshold | 0 | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 | **0.97** | 0.99 | 0.995 |
|---|---|---|---|---|---|---|---|---|---|---|
| macro-F1 | 0.681 | 0.686 | 0.728 | 0.787 | 0.814 | 0.844 | 0.862 | **0.867** | 0.846 | 0.813 |

Two cross-checks worth noting: the arg-max value (0.681) reproduces the thesis's own
68.1% exactly, and the accuracy at arg-max (98.96%) matches `fair_comparison.json`'s
`pointAcc` for PointNet++ exactly. 0.97 is a genuine interior peak, not an endpoint.

**This threshold is selected on labelled SIMULATED data**, because real data has no
ground truth. It changes what is displayed, never the model's weights.

---

## Style

Shared across all three: DejaVu Sans, white background, no top/right spines,
thin light-gray gridlines on one axis only, bold dark title with a muted
subtitle beneath, borderless right-hand summary panel with bold headline values,
300 dpi, ~7.5 in wide (fits a full text-width LaTeX figure).

Palette is consistent and meaningful — warm orange-red `#d94f2b` for the
problem/before condition, amber `#e8a33d` for the intermediate one, blue
`#2a7fbf` for fixed/selected, green `#1f9e6e` for secondary highlights.

Line plots with direct end-of-line labels were specified for time-series
comparisons; none of the three figures ended up as a time series (see figure 1's
missing panels and figure 3's absent training history), so all comparisons are
bar-based with direct value labels and no boxed legends.
