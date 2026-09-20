# Prompt to paste into the thesis-writing session

---

I have figures for Chapter 5. You have no visibility into the engineering session that
produced them, so here is everything you need. **Nothing below is estimated or invented** —
every image is either an artefact the project already produced or a plot generated directly
from measured data, and I note the source for each.

All paths are relative to the repo root:
`C:\Users\ahmed\Desktop\pointnettrain\sim2reallidar`

## Ground rule before you write a single caption

**Evaluation in simulation and evaluation on real data are two different things and must
never be blended.** In simulation there is ground truth (Gazebo's `laser_retro` attaches a
class to every point), so reconstruction *and* classification get real numbers on a
held-out test split. On real Ouster data there is **no ground truth of any kind** — nothing
there is scored, and no accuracy, F1 or precision figure may be quoted for real data. The
real-world claim is qualitative and must read that way. If a caption could be misread as
"98.96% accuracy on the real tunnel", rewrite it.

---

## Part 1 — Chapter 5 currently has no figures after §5.4

The document has nine figures (1.1, 1.2, 1.3, 4.1, 5.1, 5.2, 5.3, 5.4, 5.5). Chapter 5 runs
to §5.7, and **§5.5 onward is entirely text** — the simulation, all three encoders, the
training results and the whole real-world evaluation are unillustrated. The images below
fill that.

## Part 2 — Imagery the project already produced

These are in `thesis_figures/existing_imagery/`, renamed so the filename states its section.
They are copies; the originals are listed so you can cite provenance.

### §5.5.1 MBUT Scenario and Recording Pipeline
`thesis_figures/existing_imagery/5.5.1_scenario_object_placement_map.png`
Original: `scenario_data/mbut_76scenarios/o_sim_env_gen/scenarios/122e86f0-8b3f-46f5-8eaa-9f6b1232c7df/scenario.png`

Floor plan of the simulated tunnel with the four object classes marked at their placement
positions. **76 of these exist, one per recording scenario** — worth stating in the caption,
because it is the clearest evidence the dataset is 76 varied layouts rather than one scene
recorded 76 times. The branching side passages visible here are the same geometry §5.4.1
calls "the tunnel's T-junction".

### §5.5.3 Validation Tooling
`thesis_figures/existing_imagery/5.5.3_playback_tool_pointnet2_sim_frame040.png`
`thesis_figures/existing_imagery/5.5.3_playback_tool_salsanext_sim_frame040.png`
Originals: `demo/payloads/players/frames/{pointnet2_sim,salsanext_sim}/0040.png`

Single frames from the playback tooling the section describes. Four stacked rows: original
range, reconstructed range, ground-truth class, predicted class. Showing two candidates
side by side evidences the "one variant per encoder candidate's own output format" claim in
one image. **This is simulated data — the ground-truth class row exists precisely because
it is simulation.**

### §5.6.2 PointNet++ Encoder Implementation
`thesis_figures/existing_imagery/5.6.2_pointnet2_range_original_vs_latent_recon.png`
Original: `outputs/pointnet2/range_heatmaps/2026_08_25_17_18_39_769_sphere.png`

Original range image vs. the latent-only reconstruction, reprojected onto the real rays.
The visual counterpart to the section's claim that the reconstruction decoder "maps the
latent vector alone … with no connection back to the encoder's intermediate layers". Also
shows the 64 × 1024 grid and the reprojection that §5.6.6's shared protocol depends on.

`thesis_figures/existing_imagery/5.6.2_pointnet2_latent_only_reconstruction_3d.png`
Original: `outputs/pointnet2/recon_viz/2026_08_25_17_18_39_769_sphere.png`

The same scan in 3D, top-down and side: ground-truth environment in grey, the object class
coloured, and the 2,048 decoded points in red. Eight of these exist across four test scans,
one per object class. These two images are the strongest available evidence that the 256-d
bottleneck actually retains corridor geometry.

### §5.6.6 Training Results
`thesis_figures/existing_imagery/5.6.6_cnn_autoencoder_reconstruction_pairs.png`
Original: `outputs/kevin_cnn/reconstructions.png`

Ten original/reconstructed range-image pairs from the CNN autoencoder. The reconstructions
are visibly smoothed. This is the direct evidence for the design claim made in §4.4.2 and
repeated in §5.6.3 — that a bottleneck-only decoder "smooths away small, sparse detail".
Currently that claim is asserted in prose and demonstrated nowhere.

### §5.7.3 Real-Time Inference Implementation
`thesis_figures/existing_imagery/5.7.3_real_bag_prediction_at_threshold_0.97.png`
Original: `demo/payloads/players/frames/bag3_pointnet2/0106.png`

A **real** Ouster OS1-64 frame with the confidence-thresholded class row. Three rows, not
four — there is no ground-truth row because real LiDAR carries no labels. The class row is
nearly empty: that is the threshold suppressing the speckle the section describes. Caption
must say the class row is the model's own prediction with nothing to check it against.

### §5.7.4 Real-World Evaluation
`thesis_figures/existing_imagery/5.7.4_sim2real_object_placement_plan.png`
`thesis_figures/existing_imagery/5.7.4_sim2real_four_bag_configurations.png`
`thesis_figures/existing_imagery/5.7.4_sim2real_tight_cluster_1m.png`
`thesis_figures/existing_imagery/5.7.4_sim2real_tight_cluster_10cm.png`
Originals: `sim2real_placement_map.png`, `sim2real_4bags_map.png`, `tight_cluster_*.png`

The real-world experimental protocol, with every dimension measured from the real drive:
sphere Ø494 mm at 20 m, cylinder Ø220×595 mm at 40 m, box1 605×390×580 mm at 60 m, box2
655×390×370 mm at 80 m, plus the recorded robot path and origin; four distinct bag
configurations (A–D); and two tight-cluster configurations with all four objects inside a
4.40 m span.

§5.7.4 currently says real-world evaluation "is underway but not complete" and shows nothing
of the protocol. These four images document a carefully measured experimental design. They
turn "not finished" into "designed, executed, analysis pending" — which is a much stronger
position, and an honest one.

## Part 3 — Figures generated from measured data

In `thesis_figures/`, all 300 dpi, ~7.5 in wide, consistent style.

| file | section | generated from |
|---|---|---|
| `fig1_drift_fix_comparison.png` | §5.3.4 | thesis Ch. 5 prose (§5.2.5, §5.3.2, §5.3.4, Fig. 5.4) |
| `fig2_reconstructed_pointcloud.png` | §5.2 / §5.4.1 | `noslil.ply` (482,757 pts, CloudCompare) |
| `fig3_encoder_comparison.png` | §5.6.6 | `outputs/measured/fair_comparison.json`, `evaluate_pointnet2.json` |
| `fig4_jetson_realtime.png` | §5.7.2 | `benchmarks/pointnet2_jetson_bench.py` values per §5.7.2 |
| `fig5_decimation_validation.png` | §5.6.4 | thesis Table 5.1 |
| `fig6_confidence_threshold.png` | §5.7.3 | `outputs/measured/evaluate_pointnet2_sweep.json` |

**fig1** extends the existing Figure 5.4, which shows only three jump counts. Adds largest
jump and trajectory extent on log axes, so a 12.2 km → 5.5 mm change is legible, plus a
per-condition summary. It states on its face that it is built from the thesis's own
aggregated statistics, not re-derived from raw logs — see Part 5.

**fig2** can sit beside the existing Figure 5.5 (the CloudCompare mesh) to show the point
cloud the mesh was reconstructed *from*. Important: it reports **two** extents, because the
raw bounding box (315.9 × 100.5 × 18.7 m) is inflated by 2,591 stray SLAM points, while 99%
of the cloud spans 124.9 × 34.6 × 6.1 m. **Do not quote the 316 m figure alone.**

**fig3** goes beside Table 5.2 — same numbers, and it also shows the CNN autoencoder as
"no classifier" rather than a zero bar.

**fig4** turns §5.7.2's Jetson sentence into a figure: latency misses the sensor's 10 Hz by
54×, memory goes 156 MB → 14 GB across the same sweep.

**fig5** turns Table 5.1's sixteen raw counts into grouped bars, so "thin azimuth, never
rings" is visible rather than inferred.

**fig6** is the measurement that justifies the confidence threshold — see Part 4.

## Part 4 — A content gap you should fix in the text, not just illustrate

**§5.6.6 currently reports the classifier at raw arg-max only**: macro-F1 68.1%, per-class
precision 33.6–71.4%. That is correct and matches `evaluate.py` at threshold 0 exactly.

But §5.7.3 then introduces a confidence threshold without stating its value or justifying
it, and §5.7.4 says it "was implemented the same day this evaluation was conducted and has
not yet been independently validated".

**It has now been measured.** Sweeping the threshold against the labelled **simulated**
test set (1,140 held-out samples, one shared forward pass) gives:

| threshold | 0 (arg-max) | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 | **0.97** | 0.99 | 0.995 |
|---|---|---|---|---|---|---|---|---|---|---|
| macro-F1 | 0.681 | 0.686 | 0.728 | 0.787 | 0.814 | 0.844 | 0.862 | **0.867** | 0.846 | 0.813 |

Two things worth saying in the text. First, **the arg-max value reproduces the thesis's own
68.1% exactly**, which means the sweep is measuring the same quantity §5.6.6 already
reports — it is an extension of that result, not a competing one. Second, **0.97 is a real
peak, not an endpoint**: macro-F1 declines at 0.99 and 0.995, so the value was selected
rather than pushed as high as it would go. Cylinder, the weakest class, goes from 33.6% to
78.1% precision across the same sweep, at a cost of roughly 10.7 points of mean recall.
`fig6` is that sweep.

Please update the text accordingly:
- §5.6.6 should report both the arg-max numbers **and** the thresholded ones, clearly
  labelled as two operating points of the same model — the threshold changes what is
  displayed, never the weights.
- §5.7.3 should state the value (0.97) and that it was selected by sweeping macro-F1.
- §5.7.4's "not yet independently validated" should become: validated on simulated data,
  where ground truth exists; **still qualitative on real data, where it does not**. That
  distinction is the whole point — do not let the 86.7% leak into a sentence about the real
  tunnel.

## Part 5 — What does NOT exist, so you don't plan around it

The `jackal-tunnel-slam` / `jackal-slam-to-mesh` repository is **not on this machine**
(searched C:, D:, E:). There are no per-message diagnostic CSVs, no `odom_covariance_debug.py`,
no `tools/diag_*.py`. Consequently there is **no** per-step jump-magnitude time series, **no**
before/after X–Y trajectory plot, and **no** jump-vs-rotation-rate scatter, and none should
be requested or described as if they exist. fig1 is summary statistics only and says so.

Separately: **no per-epoch training curves exist for any candidate.** `train_utils.fit()`
hands each epoch to a `log_fn` that only prints to stdout, and every
`outputs/*/train_metrics.json` stores final scalars alone. There is no `.log` or `.csv`
capture anywhere. So do not write "Figure X shows the training loss over epochs" — that
figure cannot be produced from this project's artefacts.

## Part 6 — LaTeX

A two-column-width figure, e.g.:

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/fig5_decimation_validation.png}
  \caption[Per-class frame coverage by decimation pattern]{%
    Per-class frame coverage for each decimation pattern tested, as a percentage of the
    600 recorded frames. Keeping full ring resolution and thinning only azimuth
    (32{,}768 points) retains substantially more of every object class than either
    16{,}384-point alternative. Cylinder remains the weakest class throughout, a
    limitation of the recording rather than of the decimation strategy.
    Measured values from Table~\ref{tab:decimation}.}
  \label{fig:decimation}
\end{figure}
```

Copy the PNGs into whatever `figures/` directory the document uses. The tall
portrait-format images (§5.5.1 scenario map, §5.7.4 placement maps) will want
`[height=0.8\textheight]` or a `sidewaysfigure` rather than `width=\textwidth`.

## Part 7 — Also available if you want them

- `MBUTMESH/MBUTMESH.ply` — the actual CloudCompare Poisson mesh (841,234 vertices /
  1,680,544 faces). Could be rendered beside fig2 to show the point-cloud → mesh step
  in §5.4.1.
- `Mesh[25.ply` — an earlier, smaller reconstruction (248,094 verts, July). Could
  illustrate §5.4.4's "iterated on by hand repeatedly for each new recording".
- `real_bags/bag3_salsanext_finetuned_fixed.mp4` — **this is the video** §5.7.4 describes
  as "visibly reduced false-positive speckle … observed by eye, on video". A before/after
  still pair could be extracted from it.
- The other 75 scenario floor plans, if a grid of several layouts would strengthen §5.5.1.

Tell me which of these you want and I will produce them.
