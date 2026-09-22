# Corrections and additions for the thesis

Everything below is measured, with the command that produced it. Ordered by how
much it changes what the thesis claims.

---

## 1. §5.6.6 — "classification is a near-tie" is wrong

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

## 2. Limitations — the split claim is wrong, in the direction that costs you

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

## 3. Factual: the corridor is not 300 m

Stated as *"approximately 1.95 m wide and 300 m long"* in three places.

**Measured:**
- every one of the 76 simulated drives is **~121 m** (median 121.4 m, range 119.5-124.0)
- the point cloud those drives were built from spans **124.9 m** (the thesis's own §5.3.1)
- the real-world placement maps top out at 100 m
- bag3's real drive is 165.9 m of *path*, with 84.6 m straight-line displacement

The mesh is built from a 124.9 m cloud and cannot yield a 300 m simulated
corridor. If 300 m is the physical tunnel of which ~125 m was reconstructed, say
that — it is a fine statement and it strengthens the "one tunnel" limitation.

## 4. Factual: the 3.4 m corridor width

§5.3.3 says *"In a corridor measured at 3.4 m wide"*; elsewhere it is 1.95 m.

**Measured from the point cloud** (waist-height slab, 24 cross-sections along
the main axis): median **1.92 m**, p25 1.87, p75 1.97. The 1.95 m figure is
right for the main corridor. Either name the wider section the 3.4 m refers to,
or correct it — as written, two passes 2.19 m apart cannot fit.
