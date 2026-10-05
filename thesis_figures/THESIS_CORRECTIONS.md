# Corrections and additions for the thesis

Everything below is measured, with the command that produced it. Ordered by how
much it changes what the thesis claims.

---

## 1. §5.7.3 / §5.7.4 — every real-bag prediction so far came from a mis-fed model

**The problem:** the real Ouster and the simulator produce the same 64 x 1024 grid
*shape* but not the same *order*. The Ouster's row 0 is its highest beam and its
columns run clockwise from 0°; the simulator's row 0 is its lowest beam and its
columns run counter-clockwise from -180°. PointNet++'s ball query keeps the first
32 in-range points **by array index**, so the order of the points is part of the
input. Real scans were fed in Ouster order, and unpadded (28-32k points instead
of the 32,768 every training sample had). Nothing crashed; the model just saw
neighbourhoods it was never trained on.

**Measured — on the labelled simulated test split, fed as the real sensor delivers
it** (1,140 scans, epoch39, threshold 0.97):

| input | macro-F1 | sphere | cylinder | box2 | box1 |
|---|---|---|---|---|---|
| simulator order, training input path | 0.867 | 0.875 | 0.794 | 0.881 | 0.917 |
| Ouster order, old input path | **0.174** | 0.231 | 0.124 | 0.207 | 0.132 |
| Ouster order + measured real dropout, old path | 0.183 | 0.235 | 0.131 | 0.225 | 0.141 |
| Ouster order + measured real dropout, **fixed path** | **0.869** | 0.889 | 0.807 | 0.876 | 0.903 |

The first row reproduces `evaluate.py` exactly (0.867 / 0.681 at arg-max). The
fixed path reorders each real scan into simulator order, drops returns outside
the simulated sensor's 0.30-120 m range, and pads to 32,768 points with the very
function training uses. With it, the measured real dropout profile costs
nothing at 0.97 (0.869 vs 0.867).

**What this changes in the text:**
- **§5.7.3's figure** (`existing_imagery/5.7.3_real_bag_prediction_at_threshold_0.97.png`,
  from the demo's bag3 player) shows the broken input. Its nearly empty class row
  is not "the threshold suppressing speckle" — it is a model scoring ~0.17
  macro-F1-equivalent. Replace it with a frame from the fixed run
  (`outputs/real_inference/bag3_20260910_152806/inference.mp4`).
- **§5.7.4's real-data observations** (speckle, where objects appear) were made
  on that broken input and should be redone from the fixed runs. The earlier
  finding that predictions did not line up with the planned object positions
  was made on it too.
- **The noise fine-tune conclusion** ("no clear gain") is doubly unsupported: it
  was judged on real scans fed in the wrong order, and the ring-noise version
  was trained with the measured dropout profile applied upside down
  (`real_noise_augment.py`, fixed 2026-09-28).
- **What the fixed run shows on the real bags** (every 4th scan, threshold 0.97,
  `real_eval/before_after.py bags`; no ground truth, so these measure how
  *coherent* the predictions are, never whether they are *right*):

  | | bag1 before → after | bag2 before → after | bag3 before → after |
  |---|---|---|---|
  | object points per scan | 75 → 140 | 81 → 141 | 67 → 177 |
  | isolated in their own scan | 4.6% → 3.2% | 3.6% → 2.5% | 5.8% → 3.0% |
  | near-field object points confirmed by ≥ 8 scans | 47% → 65% | 85% → 91% | 43% → 70% |
  | persistent clusters | 35 → 30 | 32 → 23 | 34 → 25 |
  | points in a cluster sharing its majority class | 91% → 74% | 91% → 79% | 89% → 81% |
  | recon range MSE (normalised) | 0.00068 → 0.00085 | 0.00054 → 0.00068 | 0.00078 → 0.00095 |

  Predictions become more coherent: fewer isolated specks, more of them
  repeated from pose to pose. Three results cut the other way and must be
  reported with it:
  1. **61-90% of the persistent object points** (bag1 62%, bag2 90%, bag3 61%)
     sit in clusters wider than 0.9 m — wider than any placed object can span
     at a 0.3 m voxel. Each is either several objects close together (the
     tight-cluster layouts would merge) or real structure labelled as an
     object. Nothing on real data can say which.
  2. **Class consistency within a cluster falls** (~90% → 74-81%): the fixed
     model agrees *that* something is there more than *what* it is.
  3. **Real reconstruction error rises ~25%**, although on simulated scans fed
     in Ouster order the fixed path *halves* it (range MSE 0.00053 → 0.00025,
     Chamfer 0.59 → 0.37 over 95 test scans). Not explained; one candidate is
     that the fixed model reconstructs a simulator-like scene, floor 9 cm lower
     than the real one.

  So the input bug is fixed and proven fixed; what remains on real data is
  a genuine sim-to-real gap (uneven real beam spacing, real clutter the
  simulated corridor lacks) that input handling cannot close.
  Maps: `outputs/real_inference/<bag>/objects_map.png` (after) and
  `<bag>_legacy/objects_map.png` (before); figures
  `outputs/real_inference/before_after_metrics.png` and `before_after_frames.png`.
- **SalsaNext and Kevin's CNN** still receive real scans in Ouster order —
  for these image models that is upside down and mirrored — so the real-data
  comparison between candidates (e.g. PointNet++ ~35x better than SalsaNext on
  bag3 reconstruction) is not yet a fair one. Not fixed in this pass.

Commands:
```
python real_eval/before_after.py sim      # the table above
python ros2_inference/run_inference_on_bag.py --bag <bag>
python real_eval/detect_persistent_objects.py outputs/real_inference/<bag>
```
Figure: `outputs/measured/preprocessing_ablation.png`

---

## 2. §5.6.6 — "classification is a near-tie" is wrong

**Current text:** *"classification performance is a near-tie between PointNet++
and SalsaNext"* (98.96% vs 98.93% overall point accuracy).

**The problem:** the same paragraph argues, correctly, that overall accuracy is
dominated by the environment class — that is precisely why macro-F1 is reported
for PointNet++. A near-tie on the metric the section has just discredited is not
a claim the section can support.

**Measured (both models, same test split, one shared forward pass each):**

| threshold | PointNet++ macro-F1 | SalsaNext macro-F1 |
|---|---|---|
| 0 (arg-max) | 0.681 | 0.677 |
| 0.70 | 0.787 | 0.823 |
| 0.90 | 0.844 | 0.875 |
| **0.97** | **0.867** | **0.886** |
| 0.99 | 0.846 | 0.856 |

Per class at 0.97 (F1):

| class | PointNet++ | SalsaNext |
|---|---|---|
| Sphere | 0.875 | 0.885 |
| **Cylinder** | **0.794** | **0.854** |
| Box 2 | 0.881 | 0.883 |
| Box 1 | 0.917 | 0.922 |

**What to write instead:** at raw arg-max the two are genuinely within 0.4 F1
points (0.681 vs 0.677) — so "near-tie" is defensible *there*, and only there.
Once the confidence threshold is applied, SalsaNext leads on every object class
and by 1.9 macro-F1 points overall, with the largest margin (6.0 points) on
cylinder, the weakest class for both models.

**This does not overturn the thesis's main result.** PointNet++ still wins
reconstruction by ~3.9x over the CNN autoencoder and ~7.5x over SalsaNext. The
refined conclusion is stronger and more interesting than the current one:

> PointNet++ is the better reconstructor; SalsaNext is the better classifier.
> Which candidate to deploy therefore depends on what the downstream policy
> consumes — and since only the encoder ships, reconstruction quality is the
> property that constrains the latent, which is why PointNet++ remains the
> selected candidate.

Command:
```
python kevin_pipeline/fair_comparison.py --json-out outputs/measured/fair_comparison.json
```
Figure: `thesis_figures/fig7_classifier_head_to_head.png`

---

## 3. Limitations — the split claim is wrong, in the direction that costs you

**Current text:** *"the test result measures generalization to unseen stretches
and viewpoints of the same tunnel, not to unseen scenarios; an object may be
seen in blocks of more than one set."*

**Measured from `data/sim/*_ids.npy` against `poses.csv`:**

| scenarios contributing to | count |
|---|---|
| train only | 38 |
| train + val | 19 |
| test + val | 19 |
| **train + test** | **0** |

**No scenario contributes to both train and test.** The test split is 19
entirely held-out scenarios — object placements never seen during training. The
result is stronger than the thesis claims.

**But there is a leak the thesis does not mention:** val and test share all 19
scenarios, so early stopping selected the checkpoint using data from the same
scenarios the test numbers come from. It is mild — the nearest val frame to a
test frame is a median of 39.0 m away, and only 3.3% are within 3 m — but it
should be stated.

**What to write instead:** the test split holds out whole scenarios, so it does
measure generalization to unseen object placements within this tunnel; model
selection, however, used validation frames drawn from those same scenarios, so
the test set is not fully independent of the checkpoint choice.

---

## 4. Factual: the corridor is not 300 m

Stated as *"approximately 1.95 m wide and 300 m long"* in three places.

**Measured:**
- every one of the 76 simulated drives is **~121 m** (median 121.4 m, range 119.5-124.0)
- the point cloud those drives were built from spans **124.9 m** (the thesis's own §5.3.1)
- the real-world placement maps top out at 100 m
- bag3's real drive is 165.9 m of *path*, with 84.6 m straight-line displacement

The mesh is built from a 124.9 m cloud and cannot yield a 300 m simulated
corridor. If 300 m is the physical tunnel of which ~125 m was reconstructed, say
that — it is a fine statement and it strengthens the "one tunnel" limitation.

## 5. Factual: the 3.4 m corridor width

§5.3.3 says *"In a corridor measured at 3.4 m wide"*; elsewhere it is 1.95 m.

**Measured from the point cloud** (waist-height slab, 24 cross-sections along
the main axis): median **1.92 m**, p25 1.87, p75 1.97. The 1.95 m figure is
right for the main corridor. Either name the wider section the 3.4 m refers to,
or correct it — as written, two passes 2.19 m apart cannot fit.
