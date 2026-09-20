# Chapter 5 — where each image belongs

The thesis currently has **nine** figures (1.1, 1.2, 1.3, 4.1, 5.1, 5.2, 5.3, 5.4, 5.5).
Chapter 5 runs to §5.7 and **everything from §5.5 onward is text-only** — the simulation,
all three encoders, the training results and the entire real-world evaluation have no
imagery at all, despite the project having produced plenty.

Two groups below: **imagery that already exists** (staged in `existing_imagery/` with
section-named copies, ready to `\includegraphics`), and **figures generated** from real
project data (`fig*.png`).

---

## A. Existing project imagery → thesis section

### §5.5.1 MBUT Scenario and Recording Pipeline
**`existing_imagery/5.5.1_scenario_object_placement_map.png`**
Source: `scenario_data/mbut_76scenarios/o_sim_env_gen/scenarios/<id>/scenario.png`
(one per scenario — **76 of them**; the one staged is `122e86f0…`, the scenario used in
every playback and in figure 3's player).

The section describes "a corridor approximately 1.95 m wide and 300 m long, populated
with the four object classes" — and shows nothing. This is the floor plan of that
corridor with the object placements marked. It also makes the branching side-passage
geometry concrete, which §5.4.1 refers to as "the tunnel's T-junction".

> Worth saying in the caption that 76 of these exist, one per scenario — it is the most
> direct evidence that the dataset is 76 varied layouts rather than one scene recorded
> 76 times.

### §5.5.3 Validation Tooling
**`existing_imagery/5.5.3_playback_tool_pointnet2_sim_frame040.png`**
**`existing_imagery/5.5.3_playback_tool_salsanext_sim_frame040.png`**
Source: `demo/payloads/players/frames/{pointnet2_sim,salsanext_sim}/0040.png`

The section describes "a family of frame-by-frame playback tools … one variant per
encoder candidate's own output format". These are single frames from exactly that tool:
four stacked rows — original range, reconstructed range, ground-truth class, predicted
class. Showing two candidates side by side demonstrates the "one variant per candidate"
claim in one image.

### §5.6.2 PointNet++ Encoder Implementation
**`existing_imagery/5.6.2_pointnet2_range_original_vs_latent_recon.png`**
Source: `outputs/pointnet2/range_heatmaps/2026_08_25_17_18_39_769_sphere.png`

Original range image vs. the latent-only reconstruction reprojected onto the real rays.
This is the visual counterpart to the section's claim that the reconstruction decoder
"maps the latent vector alone … with no connection back to the encoder's intermediate
layers". It also shows the 64 × 1024 grid and the reprojection that §5.6.6's shared
protocol depends on.

**`existing_imagery/5.6.2_pointnet2_latent_only_reconstruction_3d.png`**
Source: `outputs/pointnet2/recon_viz/2026_08_25_17_18_39_769_sphere.png`

The same scan in 3D — top-down and side — with ground-truth environment in grey, the
object class coloured, and the decoded 2,048 points in red. Eight of these exist, one per
object class across four test scans.

> These two are the strongest argument in the chapter that the 256-d bottleneck actually
> retains the corridor geometry, and neither is currently in the document.

### §5.6.6 Training Results
**`existing_imagery/5.6.6_cnn_autoencoder_reconstruction_pairs.png`**
Source: `outputs/kevin_cnn/reconstructions.png`

Ten original/reconstructed range-image pairs from the CNN autoencoder. The reconstructions
are visibly smoothed — which is the direct evidence for the design claim (§4.4.2, and
repeated in §5.6.3) that a bottleneck-only decoder "smooths away small, sparse detail".
Right now that claim is asserted in prose and shown nowhere.

### §5.7.3 Real-Time Inference Implementation
**`existing_imagery/5.7.3_real_bag_prediction_at_threshold_0.97.png`**
Source: `demo/payloads/players/frames/bag3_pointnet2/0106.png`

A real Ouster frame with the confidence-thresholded class row. Note the class row is
nearly empty — that is the threshold suppressing the speckle the section describes. Pair
it with **fig6** below, which is the measurement that justifies the threshold value.

### §5.7.4 Real-World Evaluation
**`existing_imagery/5.7.4_sim2real_object_placement_plan.png`** — the real-world
experimental setup: sphere Ø494 mm at 20 m, cylinder Ø220×595 mm at 40 m, box1
605×390×580 mm at 60 m, box2 655×390×370 mm at 80 m, with the recorded robot path and
origin. Every dimension measured from the real drive.

**`existing_imagery/5.7.4_sim2real_four_bag_configurations.png`** — the four distinct bag
configurations (A–D), each with different object types and distances along the same path.

**`existing_imagery/5.7.4_sim2real_tight_cluster_1m.png`** and **`…_10cm.png`** — the
two tight-cluster configurations, all four objects within a 4.40 m span.

> §5.7.4 currently says real-world evaluation "is underway but not complete" and shows
> nothing of the protocol. These four images document a carefully measured experimental
> design, which is worth showing even while the results are in progress — they turn "not
> finished" into "designed, executed, analysis pending".

---

## B. Figures generated from real data

| file | section | source |
|---|---|---|
| `fig1_drift_fix_comparison.png` | §5.3.4 (replaces/extends the existing Fig. 5.4) | thesis Ch. 5 prose |
| `fig2_reconstructed_pointcloud.png` | §5.2 / §5.4.1 (beside the existing Fig. 5.5) | `noslil.ply` |
| `fig3_encoder_comparison.png` | §5.6.6 (beside Table 5.2) | `outputs/measured/*.json` |
| `fig4_jetson_realtime.png` | §5.7.2 | `benchmarks/pointnet2_jetson_bench.py` values, thesis §5.7.2 |
| `fig5_decimation_validation.png` | §5.6.4 (replaces/illustrates Table 5.1) | thesis Table 5.1 |
| `fig6_confidence_threshold.png` | §5.7.3 | `outputs/measured/evaluate_pointnet2_sweep.json` |

**fig1** — the existing Figure 5.4 shows only three jump counts. This adds largest jump,
trajectory extent and a per-condition summary, on log axes that make a 12.2 km → 5.5 mm
change legible.

**fig4** — §5.7.2 reports the Jetson numbers as a sentence. As a figure the argument is
immediate: latency crosses the 10 Hz line by 54×, and memory goes from 156 MB to 14 GB
across the same sweep.

**fig5** — Table 5.1 is four configurations × four classes as raw counts out of 600. As
grouped bars the conclusion ("thin azimuth, never rings") is visible instead of inferred.

**fig6** — the section says a threshold is applied but gives no value and no
justification; §5.6.6 still reports only the un-thresholded macro-F1 of 68.1%. See the
content gap below.

---

## C. Content gap worth fixing in the text, not just the figures

**§5.6.6 reports the classifier at raw arg-max only** — macro-F1 68.1%, per-class
precision 33.6–71.4%. That matches `evaluate.py` at threshold 0 exactly. But §5.7.3 then
introduces a confidence threshold and §5.7.4 says it "was implemented the same day … and
has not yet been independently validated".

It has now been measured, in one shared forward pass over the 1,140-sample simulated test
split:

| threshold | 0 (arg-max) | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 | **0.97** | 0.99 | 0.995 |
|---|---|---|---|---|---|---|---|---|---|---|
| macro-F1 | 0.681 | 0.686 | 0.728 | 0.787 | 0.814 | 0.844 | 0.862 | **0.867** | 0.846 | 0.813 |

The arg-max value reproduces the thesis's own 68.1% exactly, so this extends §5.6.6's
result rather than competing with it. And 0.97 is a genuine peak — macro-F1 falls at 0.99
and 0.995 — so the threshold was selected, not pushed as far as it would go. Cylinder
precision goes 33.6% → 78.1% across the sweep, costing ~10.7 points of mean recall.

This changes §5.7.4's framing from "adjustment not yet validated" to "validated on
simulated data, where ground truth exists; still qualitative on real data, where it does
not".

`fig6_confidence_threshold.png` is that measurement.

---

## D. Not used, but available if you want them

- `MBUTMESH/MBUTMESH.ply` — the actual CloudCompare Poisson mesh, 841,234 verts /
  1,680,544 faces. Could be rendered beside fig2 to show the cloud → mesh step (§5.4.1).
- `Mesh[25.ply` — an earlier, smaller reconstruction (248,094 verts), 2026-07-21. Could
  illustrate §5.4.4's "iterated on by hand repeatedly for each new recording".
- `scenario_data/twc_scenario_6633e5c2/environment/mbut-mesh-ahmed/occupancy.png` — the
  Gazebo occupancy grid derived from the mesh.
- `real_bags/bag3_salsanext_finetuned_fixed.mp4` (159 MB) — the SalsaNext
  domain-randomization fine-tune on real data that §5.7.4 describes as "visibly reduced
  false-positive speckle … observed by eye, on video". This *is* that video. A
  before/after still pair could be extracted from it.
- The other 75 `scenario.png` files, if a grid of several layouts would help §5.5.1.
