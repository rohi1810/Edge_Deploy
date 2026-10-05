# sim2real-lidar

Compress a LiDAR scan into a compact latent vector a downstream reinforcement-learning
policy can consume as its observation — and find out whether an encoder trained
entirely in simulation still behaves sensibly on a real sensor.

Three candidate encoders share one contract, `encode(scan) -> (B, 256)`, are trained on
the same simulated dataset, and are scored on the same held-out split:

| candidate | representation | reconstruction | classification | where |
|---|---|---|---|---|
| **PointNet++** | raw 3D points | Chamfer, free-form 2,048 points | per-point, 8-class head | `models/pointnet2.py` |
| **CNN autoencoder** | `(64,1024)` range image | native grid | **none** — never had one | sibling repo, loaded via `viz/gen_playback_kevin.py` |
| **SalsaNext** | `(64,1024)` range image | native grid | per-pixel, 8-class head | `models/salsanext.py` |

PointNet++ is the one this repo is built around and the one that deploys. The other two
are the comparison it has to beat.

> **Simulation and real data are evaluated differently, and the distinction matters.**
> In simulation there are ground-truth labels, so reconstruction *and* classification
> get real numbers on a held-out test split. On real data there is **no ground truth at
> all** — a real Ouster carries no `laser_retro` — so nothing there is scored. The
> real-world claim is qualitative, and the demo says so on the page.

---

## Quick start

```bash
pip install -r requirements.txt          # training machine
python3 -m pytest tests/ -v              # synthetic fixtures, no data or GPU needed
```

To just look at the results, open `demo/index.html` in a browser. It needs no server,
no network and no Python — see [The demo](#the-demo).

## The data

**76 recorded Gazebo scenarios, 7,600 scans**, of a Clearpath Jackal driving the "MBUT"
tunnel corridor (~1.95 m wide, ~300 m long). The simulated world is built from a real
mesh reconstruction of that corridor, which is what makes the sim-to-real comparison
meaningful rather than a comparison against an invented room.

Split **5,320 / 1,140 / 1,140** (train/val/test) by `data_prep/split.py`, spatial-block
so adjacent near-duplicate frames never land in different splits. Every candidate reads
the same `data/sim/{train,val,test}_ids.npy`.

Each scan is 65,536 points (`x, y, z, intensity, ring`) — a 64-ring × 1024-column
organized scan — with a per-point class label carried in `intensity` via Gazebo's
`laser_retro`:

| raw value | class | present? |
|---|---|---|
| 0.0 | environment | 99.10% of all points, **and** no-return points |
| 1.0 / 2.0 / 3.0 | human / car / bus | **zero examples** |
| 4.0 | sphere | 0.24% |
| 5.0 | cylinder | 0.09% — the rarest present class |
| 6.0 | box2 | 0.17% |
| 7.0 | box1 | 0.40% |

Human/car/bus are not rare, they are **absent**. No loss weighting manufactures signal
that does not exist. The class head is sized for the full 8-value taxonomy so a future
scenario needs no architecture change, but treat those three as untested, not as
"probably fine".

**Two gotchas that silently break things:**

1. **Class labels are not exact integers.** The GPU ray tracer yields `6.98863`, not
   `7.0`. Always `np.rint()` before comparing — `dataset.py::_load_raw` does. A naive
   `intensity == 7.0` misses almost every object point.
2. **No-return points (`inf` in x/y/z) also carry `intensity == 0.0`**, identical to an
   environment hit. Filter on `isfinite`, never on class value, or they poison every
   distance computation downstream (FPS, ball query, Chamfer) with NaN/inf.

### Why raw points, not the range grid

The `(64,1024)` rasterization is **not lossy for this sensor**: the `ring` field is
already in clean ring-major order, so all 65,536 points land in 65,536 distinct cells.
It is a lossless reindexing.

What raw points buy is different: explicit 3D coordinates handed to the network, rather
than a scalar range the network must implicitly combine with grid position to recover
geometry — and no commitment to convolution's fixed-topology assumption. What they
cost is that PointNet++ does not scale with point count as gracefully as a 2D
convolution, which is why a point *budget* is a real constraint for this candidate and
not for the other two.

`data_prep/parse_recordings.py` stores a scalar **range** per pixel, not x/y/z. That is
one-way (inverting it needs each ray's exact beam angle, which is not recorded), so
`dataset.py` reads the original raw recordings instead — a separate data path from the
range-image candidates, joined only by sharing `sample_id` keys and the split.

## Point budget and the Jetson

**Deployment target: NVIDIA Jetson AGX Orin 64GB, JetPack 5.1.x, inference only.**
Training happens on a separate machine (Quadro RTX 5000).

65,536 points/scan is too many for PointNet++'s FPS + ball-query hierarchy on Orin. The
fix is to **decimate the ray pattern itself** — keep every ring, keep every 2nd azimuth
column — rather than randomly dropping points or voxel-downsampling. That is literally
simulating a coarser sensor, and applying the identical decimation to the live feed
means no train/inference mismatch in *which points the model even sees*.

### Why azimuth, never rings

Measured, not assumed — object-class frame coverage at equal point counts:

| decimation | points | sphere | cylinder | box2 | box1 |
|---|---|---|---|---|---|
| every 2nd ring, every 2nd col | 16,384 | 481 | 271 | 259 | 542 |
| every ring, every 4th col | 16,384 | 424 | 222 | 413 | 550 |
| **every ring, every 2nd col** | **32,768** | **528 (91%)** | **316 (77%)** | **501 (96%)** | **598 (100%)** |

Same point count, very different results depending which axis gets cut. With only 64
rings to begin with, small objects span a handful of them, so vertical decimation risks
skipping objects entirely; azimuth starts at 1024 and is far finer. Implemented as
`RING_STRIDE = 1, COL_STRIDE = 2` in `dataset.py` → `64 × 512 = 32,768` points.

Uniform decimation still does not fix class imbalance (cylinder coverage drops anyway).
A class-aware scheme that guarantees object points survive was considered and
**rejected**: it needs ground-truth labels to decide what to keep, which do not exist at
inference time — exactly the mismatch this design avoids.

### Measured cost on the real Orin

`benchmarks/pointnet2_jetson_bench.py`, `PointNet2Encoder`, single-scan batches:

| points | latency | rate | peak memory |
|---|---|---|---|
| 2,048 | 242 ms | 4.1 Hz | 156 MB |
| 4,096 | 470 ms | 2.1 Hz | 156 MB |
| 8,192 | 1015 ms | 1.0 Hz | 881 MB |
| 16,384 | 2260 ms | 0.4 Hz | 3.5 GB |
| 32,768 | 5378 ms | 0.2 Hz | 14.0 GB |

None reach the sensor's native 10 Hz. That turned out not to bind: the SLAM stack
(KISS-Matcher + spark_fast_lio) is CPU-only, needs every scan for odometry, and does
not consume the classifier's output — the two need not run at the same rate. The robot
moves at ~0.38 m/s median, so 5.4 s between predictions is ~2 m of travel in a ~300 m
corridor. Workable, worth watching.

> **Caveat:** these numbers predate the `ball_query` / `farthest_point_sample` fixes
> (see [Testing](#testing)). The fixes replace a masked assignment with a `torch.where`,
> not a different algorithm, so timing should be unchanged — but "should" is not
> "confirmed". Re-run before quoting as final.

## What actually deploys — only the encoder

The decoders exist purely to shape the encoder during training (the Information
Bottleneck idea: constrain what the latent captures, push it to capture what is
predictive) and are discarded before deployment. Only `encoder.state_dict()` is ever
loaded for inference.

This is verified, not asserted:
`tests/test_pointnet2_model.py::test_only_encoder_state_dict_is_needed_for_deployment`
loads only the encoder's weights into a fresh, decoder-less `PointNet2Encoder` and
checks it reproduces `encode()` exactly.

Consequence: the decoders' structure has **zero effect on the deployment latency numbers
above**. It only costs training time, on the Quadro, which has headroom — which is why
the two decoders are free to be structured completely differently.

### Deploying the trained encoder

The repo ships one trained artifact: **`weights/pointnet2_epoch39_encoder.pt`** (1.4 MB,
352,288 parameters), the encoder of the epoch-39 PointNet++ used for every result in the
thesis. It is byte-for-byte the encoder inside the full epoch-39 multitask model, and
`tests/test_shipped_weights.py` checks that it loads into a bare `PointNet2Encoder` and
encodes a real-format scan to one finite 256-d latent. Everything else (full multitask
models, fine-tunes) stays out of git; `train.py` regenerates it.

On the edge device (see [Jetson AGX Orin](#jetson-agx-orin-inference-only) for installing
PyTorch there), run from the repo root:

```python
import sys
import torch
sys.path.insert(0, "ros2_inference")
from _shared_inference import real_scan_to_points
from models.pointnet2 import PointNet2Encoder
from sensor import check_sim_order, ouster_to_sim_order

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = PointNet2Encoder()
encoder.load_state_dict(torch.load("weights/pointnet2_epoch39_encoder.pt",
                                   map_location="cpu", weights_only=True))
encoder.to(device).eval()

# xyz_grid: one Ouster scan as the driver publishes it, (64, 1024, 3) float32
check_sim_order(ouster_to_sim_order(xyz_grid))   # once per stream: refuses a misordered driver
points, _ = real_scan_to_points(xyz_grid)        # (32768, 3), exactly what training fed the model
with torch.no_grad():
    latent, _ = encoder(torch.from_numpy(points).unsqueeze(0).to(device))   # (1, 256)
```

Two things are easy to get wrong:

- **Never feed the raw Ouster grid to the encoder.** Its rows and columns run in a
  different order from the simulator's, and PointNet++ builds its neighbourhoods from
  array position, so a raw scan silently gives a different (worse) latent.
  `real_scan_to_points` reorders, drops returns outside the simulated sensor's range and
  pads to 32,768 points. See
  [Real scans must reach the model the way simulated ones did](#real-scans-must-reach-the-model-the-way-simulated-ones-did).
- **The encoder alone does not classify.** It outputs the 256-d observation for the
  downstream policy. The per-point classes in the inference videos come from the
  segmentation decoder, which is only in the full multitask model
  (`outputs/pointnet2_checkpoints/epoch39_multitask_model.pt`, not in git).

Latency on the Orin: `benchmarks/pointnet2_jetson_bench.py`, numbers in
[Measured cost on the real Orin](#measured-cost-on-the-real-orin).

### Why the two decoders differ structurally

**`ReconstructionDecoder`** takes the latent `z` **alone** — no path back to the input.
That is what makes reconstruction quality a real test of the bottleneck: with skip
connections it could reconstruct well regardless of how good the latent is, making the
metric meaningless for its purpose. Verified by
`test_reconstruction_decoder_depends_only_on_latent_not_raw_input`. Scored with
**Chamfer distance**, since point sets have no natural pixel-wise loss.

**`SegmentationDecoder`** is the standard feature-propagation decoder *with* skip
connections to every encoder level. Its job is not to test the bottleneck but to give
the encoder a semantically rich training signal. A bottleneck-only decoder demonstrably
smooths away small sparse detail — and small objects are the classification target.
Scored with **class-weighted cross-entropy**, weights as capped inverse frequency:
capped in both directions, because the rarest present class would otherwise get a
destabilising weight, and the *absent* classes would divide by zero and produce `inf`.

## Comparing the three candidates

**`kevin_pipeline/fair_comparison.py` is the only script that puts all three on one
scale**, and its output is the only thing the demo's table may be built from.

The trap it exists to avoid: the per-run `outputs/*/train_metrics.json` files are *not*
comparable. PointNet++'s recon loss there is a **Chamfer distance in metres**, while
SalsaNext's and the CNN's are **normalized MSE**. Tabling them side by side is wrong by
roughly three orders of magnitude. `fair_comparison.py` reprojects PointNet++'s
free-form output onto the real ray grid and scores it with the same normalized formula
as the other two, on the same test ids, with the same `max_range`.

```bash
python3 kevin_pipeline/fair_comparison.py \
    --json-out outputs/measured/fair_comparison.json
```

Per-class classification numbers come from `evaluate.py`, which also implements the
confidence threshold described next:

```bash
python3 evaluate.py --class-conf-threshold 0.97 \
    --checkpoint outputs/pointnet2_checkpoints/epoch39_multitask_model.pt \
    --json-out outputs/measured/evaluate_pointnet2.json
```

### The confidence threshold is a display decision

A point keeps its predicted object class only if softmax confidence clears a threshold;
otherwise it reads as environment. **This does not change any weights** — it changes
what gets drawn. It exists because raw argmax scatters low-confidence object
predictions across the floor of real scans.

The value was chosen by sweeping against the labelled *simulated* test set, not by eye:

| threshold | 0.0 (argmax) | 0.70 | **0.97** |
|---|---|---|---|
| macro-F1 (object classes) | 0.681 | 0.787 | **0.867** |

Cylinder precision goes from 33.6% to 78.1% across that range; above 0.97 it declines
again. `evaluate.py --class-conf-threshold` reproduces any row of that table, and
`ros2_inference/_shared_inference.py` applies the same rule at inference.

## Sim2real: running on the real sensor

Real recordings come from an **Ouster OS1-64 on a Clearpath Jackal** over ROS2, whose
64 rings match the simulated sensor's grid directly.

```bash
# offline, over a recorded bag -- needs no ROS2 installation (uses `rosbags`);
# PointNet++ epoch39 by default. Writes outputs/real_inference/<bag>/:
# inference.mp4 (every scan, H.264), predictions.npz, summary.json
python ros2_inference/run_inference_on_bag.py --bag real_bags/bag3_20260910_152806

# where the model keeps seeing objects across scans -> objects.json, objects_map.png
python real_eval/detect_persistent_objects.py outputs/real_inference/bag3_20260910_152806

# evidence for the input fixes below: with labels (simulated test split), and label-free (the bags)
python real_eval/before_after.py sim
python real_eval/before_after.py bags

# live, on the robot, inside a ROS2 environment
ros2 run ... ros2_inference/live_inference_node.py
```

Both share `ros2_inference/_shared_inference.py`, so the offline and live paths cannot
drift apart. The offline run streams the bag (one scan in memory), skips and counts
unusable scans instead of dying on them, and fails if more than 1% are skipped.

### Real scans must reach the model the way simulated ones did

The Ouster and the simulator share the 64 x 1024 grid **shape**, not its **order**: the
Ouster's row 0 is its highest beam and its columns run clockwise from 0°; the simulator's
row 0 is its lowest beam and its columns run counter-clockwise from -180° (`sensor.py`).
PointNet++'s ball query keeps the first 32 in-range points *by array index*, so order is
part of the input. Until 2026-09-28 real scans went in in Ouster order, unpadded, at a
point count training never used. Scoring the labelled simulated test split as the real
sensor delivers it (`real_eval/before_after.py sim`, 1,140 scans, threshold 0.97):

| input | macro-F1 |
|---|---|
| simulator order, training input path | 0.867 |
| Ouster order, old input path | **0.174** |
| Ouster order + measured real dropout, old input path | 0.183 |
| Ouster order + measured real dropout, **fixed** input path | **0.869** |

The fixed path (`_shared_inference.real_scan_to_points`) reorders every real scan into
simulator order (`sensor.ouster_to_sim_order`), drops returns outside the range the
simulated sensor ever produced (0.30-120 m), and pads to exactly 32,768 points with the
same function training uses (`dataset.prepare_points`). It refuses a stream whose order
doesn't match the expected convention (`sensor.check_sim_order`). Real-bag outputs made
before this fix — the old `real_bags/*.html` players and videos, and the demo's bag3
player — show the broken input.

### Measured on real data, and one thing removed

- **Ring-dependent dropout is real and large.** Returns are lost at ~2% on the middle
  rings, rising to ~34% on the lowest, floor-grazing beam. `real_noise_augment.py` holds
  the measured 64-element rate array. Until 2026-09-28 it was applied to simulated grids
  upside down (the measured array is in Ouster row order); the ring-noise fine-tune
  (`outputs/pointnet2_ringnoise_finetuned`) was trained that way.
- **False points are not.** An earlier noise model injected spurious near-range points
  at 5%. Measuring the actual rate of isolated near-range spikes in the bags gave
  **0.065%**, so the injection was removed entirely rather than kept at a smaller value.
  It came from misreading a viridis colormap — yellow there is *far*, not near.
- **Ring order is inverted relative to simulation.** From the sensor's own metadata:
  ring 0 is `+21.6°` (the highest beam), ring 63 is `-21.2°`. For *display*, the `flipud`
  that is correct for simulated grids makes real playback upside down, so the real path
  does not flip; for the *model*, see above.
- **Sensor height differs** (~0.63 m real vs 0.535 m simulated). Classification does not
  depend on it — the encoder sees only neighbourhood-relative coordinates
  (`tests/test_real_inference.py`) — but the reconstruction branch outputs absolute
  positions, so part of the real reconstruction error is this offset.

### What cannot be claimed

There is no ground truth on real data. Nothing in this section produces an accuracy
number, and none should be quoted. The honest output is visual: the predicted class map
on a real drive, at a stated threshold, next to the reconstruction.

## The demo

`demo/` is a self-contained, pre-rendered presentation of all of the above — the tunnel
reconstruction, the three-candidate comparison, the per-class table, playbacks of all
three candidates on the same simulated drive, the real-bag run, and an interactive 3D
view of what the 256-d latent preserves.

**It works three ways**, and all three are deliberate:

```bash
# 1. as a folder -- no server, no Python, no network
open demo/index.html

# 2. served
docker compose up -d demo        # -> http://localhost:8080

# 3. regenerate it from checkpoints and data
python3 demo/build_demo.py --all
```

Nothing runs a model in the browser. Every frame is rendered offline by this repo's own
scripts and embedded; each playback states the checkpoint it came from.

The `file://` mode is what constrains the design: an opaque origin blocks `fetch()` and
ES modules, so every payload is a **classic script** that registers itself
(`DEMO.register(kind, id, payload)`) and three.js's `PLYLoader`/`PCDLoader` cannot be
used — geometry is pre-quantized into base64 and decoded to typed arrays by hand.

Adding a fourth candidate is a **data change**: one entry in `demo/sources.json`, one in
`demo/metrics_template.json`, one tab in `demo/payloads/manifest.js`. No new exporter
code.

```bash
python3 demo/build_demo.py --check     # validate; stdlib only, no torch, needs no data
python3 demo/build_demo.py --report    # per-file sizes and budget
python3 demo/build_demo.py --only sim_player:pointnet2_sim
```

`--check` asserts what the page cannot check for itself: that every telemetry field a
payload declares is present in *every* frame, that referenced frames and figures exist,
that timestamps do not go backwards, and that **no subresource loads from a remote
origin** (fonts are self-hosted; a Google Fonts `@import` is both an offline break and a
GDPR exposure for a German university page).

Missing inputs **skip, they do not fail** — checkpoints, `scenario_data/`, the bags and
the `.ply` geometry are all gitignored, so a fresh clone still builds the parts it has
and keeps previously committed payloads for the rest.

## Docker

```bash
docker compose up -d demo                    # nginx + pre-rendered payloads, no GPU
docker compose --profile pipeline build      # CUDA image, only if you want to regenerate
docker compose --profile pipeline run --rm pipeline python demo/build_demo.py --all
```

The pipeline service is **profile-gated** so `docker compose up` still works on a machine
with no NVIDIA runtime — a GPU reservation there is a hard error, not a warning.
Large data is bind-mounted at run time, never baked into the image; the root
`.dockerignore` keeps a 15.9 GB bag out of the build context.

nginx rather than `python -m http.server` for one concrete reason: the latter has no
HTTP Range support, so seeking inside the videos silently fails.

> **Not verified locally.** Docker is not installed on the development machine, so both
> images are written but unbuilt. Test on a machine that has it before relying on them.

## Setup

### Cloning

Code-only repo. `git clone` gets every `.py`, the tests, and the demo shell — not the
data. Two things must exist locally, neither in git:

- **`scenario_data/`** — the raw recordings. Synced by rsync/scp, not GitHub.
- **`data/`** — parsed grids + split. Do **not** sync; regenerate:

```bash
python3 data_prep/parse_recordings.py
python3 data_prep/split.py
```

Deterministic (no RNG in either script), confirmed by regenerating and diffing.

### Training machine

```bash
pip install -r requirements.txt            # training only
pip install -r requirements-pipeline.txt   # + demo rendering, video, mesh decimation
```

`requirements.txt` is the training minimum and is **not** enough to rebuild the demo —
that needs `pytorch_msssim`, `rosbags`, `pillow`, `matplotlib`, `imageio-ffmpeg`,
`fast-simplification` and `scipy`, which is what the pipeline file adds.

### Jetson AGX Orin (inference only)

**`pip install torch` does not work on Jetson.** PyPI wheels are x86_64; Jetson is
aarch64 with a CUDA build tied to the exact JetPack/L4T version. Confirm it first:

```bash
sudo apt-cache show nvidia-jetpack | grep -i version
```

then take the matching wheel from NVIDIA's Jetson PyTorch page, plus a matching
`torchvision` (not automatically compatible just because torch matches).

No custom CUDA extensions are used anywhere — `pointnet2_utils.py` is pure PyTorch
(`torch.cdist`, `torch.sort`, `torch.where`). Deliberate: compiling extensions against
Jetson's pinned CUDA is a well-known and avoidable source of setup pain.

## Training

```bash
python3 train.py                 # PointNet++
python3 train_salsanext.py       # SalsaNext
```

Each saves `best_encoder.pt` (the only artifact that deploys), `best_multitask_model.pt`
(kept for inspection/resume) and `train_metrics.json` to `outputs/<candidate>/`. To ship a
newly trained PointNet++, copy its `best_encoder.pt` into `weights/` (the one folder where
`.pt` files are tracked) and rerun `tests/test_shipped_weights.py`.

Checkpointing writes on **every** new best, not on a schedule — a `pip install` during a
long run once killed training and cost the intermediate progress.

## Testing

```bash
python3 -m pytest tests/ -v
```

`tests/conftest.py` builds a **synthetic** fixture rather than using the real
recordings, so the suite runs on any clone with no external data. The fixture matches
the real format exactly — same fields, same ring-major order, the same off-integer class
noise, the same inf-for-no-return convention — so passing is meaningful against the real
data's quirks.

Building the suite caught four real bugs, worth recording as evidence that this level of
testing pays:

1. **`farthest_point_sample` was non-deterministic** — a random start index per call
   meant the same scan could encode differently between two consecutive calls, even in
   `eval()` mode (which only affects Dropout/BatchNorm, not an explicit RNG call). A
   deployed robot would report a changing reading of an unchanged static scene. Fixed
   with a fixed start index.
2. **`ball_query` leaked an out-of-bounds index** when a centroid had zero neighbours in
   radius — on ordinary random data, not a contrived case. The sentinel (equal to the
   point count) flowed into the returned indices and would corrupt the next indexing op.
3. **`ball_query` crashed when N < k**, because `[:k]` on a shorter array silently
   returns fewer elements and broke shape assumptions several lines later.
4. **Every norm layer was `BatchNorm`**, which needs batch > 1 in training mode — so
   `--batch-size 1` crashed, despite being exactly what `train.py`'s own OOM message
   suggests. Unit tests had always used `batch_size=2` and missed it; an end-to-end
   smoke test caught it. Fixed by switching to `GroupNorm`.

## Known limitations

- **Human/car/bus have zero training examples.** The head supports them; nothing has
  ever taught the network to recognise them.
- **No real-world ground truth.** Real LiDAR has no `laser_retro`, so real-world
  classification cannot be scored — only inspected. A small labelled real set (a known
  object at a logged position) would be the way to change that.
- **Jetson numbers predate the `ball_query`/FPS fixes** and should be re-confirmed.
- **Docker images are unbuilt** — no Docker on the development machine.
- **`noslil.ply` is a placeholder scan** for the demo's tunnel section, kept until a
  cleaner capture exists. Replacing it is `demo/build_demo.py --only tunnel`.
- **Thesis metadata in `demo/payloads/manifest.js` is still `TODO —`** (author,
  university, department, supervisors).

## Project structure

```
sim2real-lidar/
├── data_prep/              parse_recordings.py, split.py  (shared by all candidates)
├── models/                 pointnet2.py, pointnet2_utils.py, salsanext.py
├── dataset.py              raw recordings -> decimated (xyz, class)   [PointNet++]
├── salsanext_dataset.py    range-image dataset                        [SalsaNext]
├── losses.py               Chamfer, class-weighted CE
├── train.py / train_salsanext.py
├── evaluate.py             test-split metrics, --class-conf-threshold, --json-out
├── real_noise_augment.py   measured per-ring dropout, for sim2real fine-tuning
├── kevin_pipeline/
│   └── fair_comparison.py  THE one script that puts all three on one scale
├── weights/                pointnet2_epoch39_encoder.pt -- the trained encoder that deploys
├── sensor.py               Ouster <-> simulator scan order, the order check, range limits
├── ros2_inference/
│   ├── _shared_inference.py    ModelRunner + real_scan_to_points, shared by both paths below
│   ├── run_inference_on_bag.py offline over a recorded bag (no ROS2 needed)
│   └── live_inference_node.py  on-robot ROS2 node
├── real_eval/
│   ├── detect_persistent_objects.py  objects the model keeps seeing -> top-down map
│   └── before_after.py         measured effect of the real-scan input fixes
├── viz/
│   ├── playback_common.py      build_playback_payload + shared rendering
│   └── gen_playback_{pointnet2,kevin,salsanext}.py
├── demo/                   the presentation -- see "The demo"
│   ├── build_demo.py       exporter CLI
│   ├── sources.json        WHAT to export  (the "add a model" lever)
│   ├── metrics_template.json   authored prose; numbers filled from measured JSON
│   └── schema.py           validation, stdlib only
├── benchmarks/             Jetson latency/memory
├── tests/                  synthetic fixtures, no external data
├── docker/pipeline.Dockerfile
└── docker-compose.yml
```
