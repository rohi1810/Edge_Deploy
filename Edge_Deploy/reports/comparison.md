# Deployment methods: comparison and next direction

Updated 27 October 2026 with the explicitly requested TopK-plus-chunking run.
All figures below are desktop Quadro RTX 5000 measurements.

**Recommendation: TopK plus chunk size 1024 is the strongest measured speed/memory
compromise. TopK alone remains fastest by mean latency; chunk size 128 minimizes
memory.** The combination is now implemented and checked on 20 scans.

## Latest comparison: 20 distinct scenarios

Same trained encoder, FP32, batch 1, 32768 points, two warmups and five timed
forwards per scan. All candidate latents matched the original exactly on every
selected scan; all 60 neighborhood checks per candidate also matched.

| Variant | Mean forward | Peak allocated GPU memory | Speed vs original |
|---|---:|---:|---:|
| Original | 340.52 ms | 7125.79 MB | 1.000x |
| TopK | 189.49 ms | 1757.08 MB | 1.797x |
| TopK + chunk 128 | 209.53 ms | 81.69 MB | 1.625x |
| TopK + chunk 256 | 202.95 ms | 122.26 MB | 1.678x |
| TopK + chunk 512 | 208.73 ms | 231.24 MB | 1.631x |
| TopK + chunk 1024 | 193.69 ms | 449.22 MB | 1.758x |

- **Chunk 1024:** 1.76x the original throughput and 93.70% less peak allocated
  memory. Compared with TopK alone, memory fell 74.43% with a measured 2.2%
  latency increase. The small timing difference should not be overinterpreted.
- **Chunk 128:** 81.69 MB, about 98.85% less than the original; slower than the
  other main candidates at 209.53 ms. Choose it when memory is the priority.
- **TopK alone:** fastest mean at 189.49 ms, but needs 1757.08 MB. Keep it as
  the speed reference.

The combination report contains medians, p95 values, checkpoint provenance and
all 20 scan paths: [TopK plus chunking](topk_chunked.md).

## Earlier experiments: three scans

Hardware: Quadro RTX 5000. Original trained PointNet++ encoder, batch size 1,
32768 points, 256-number output, same three recorded simulated scans.

| Method | Mean forward | Speed ratio vs paired original | Peak allocated GPU memory | Output comparison | Decision |
|---|---:|---:|---:|---|---|
| Original FP32 PyTorch | 339.69 ms* | 1.00x | 7125.79 MB | Reference | Keep as reference |
| PyTorch FP16 mixed precision | 339.89 ms | 1.004x | 7125.79 MB | Relative latent L2 0.102%-0.144% | No meaningful improvement measured |
| Chunked queries, size 128 | 357.16 ms | 0.947x | 235.53 MB | Bit-identical on selected scans | Historical memory improvement |
| PyTorch TopK queries | **186.81 ms** | **1.818x** | 1757.08 MB | Bit-identical on selected scans | Best speed in that earlier run |
| ONNX Runtime CUDA | 1392.04 ms | 0.244x | Not measured | Maximum relative latent L2 0.00224% | Slower in this implementation |
| TensorRT FP32 | 243.88 ms | 1.393x | Not measured | Maximum relative latent L2 0.05852% | Faster than original, slower than PyTorch TopK |
| TensorRT FP16 convolution compute | 265.50 ms | Excluded as a candidate | Not measured | Relative latent L2 7.72%-27.00% | **Numerically rejected** |

Speed ratio is original time divided by candidate time; higher is faster.
*339.69 ms is the latest original baseline. FP16's paired baseline was 341.25 ms
and chunking's was 338.20 ms. Ratios above use each experiment's paired baseline.

FP16 and chunking used 9 timed forwards per mode; TopK, ONNX and TensorRT used
15. Timings exclude preprocessing, data transfer, cold startup, export and
engine building. Memory figures are PyTorch peak allocated memory, not total
device usage. External runtime memory was not measured.

## What these results tell us

- TopK reduced latency by about 45% and peak allocated memory by about 75%
  relative to its paired original. It selects the smallest 32 candidate point
  indices instead of sorting every index; selected valid-neighbor ordering is
  preserved. This is a query implementation change, not model compression.
- Chunking reduced peak allocated memory by 96.7%, with a 5.6% latency increase.
  It processes fewer centroids at once without changing model weights or point count.
- TensorRT FP32 helped relative to the original, but was about 31% slower than
  PyTorch with the same TopK query rewrite. The query improvement must therefore
  be separated from any inference-runtime benefit.
- The ONNX export includes the whole encoder, with compact FPS loops and TopK
  queries. Prior profiling showed both CPU and CUDA node execution. Its slower
  result applies to this implementation; it does not establish that ONNX is
  always slower.
- The TensorRT FP16 variant changed the output too much even after additional
  precision constraints. Its cause has not been isolated. Do not select it for
  deployment from these results.

## Next direction

1. Keep TopK alone and combined chunk size 1024 as the principal candidates.
   Keep chunk size 128 as the low-memory option. The combined method requires
   no retraining and leaves Ahmed's original source and weights unchanged.
2. Before adding INT8 or pruning, profile the selected variant to identify the
   remaining bottleneck. These results establish a query implementation benefit;
   they do not establish that weight compression will resolve the remaining cost.
3. Evaluate prediction quality with the matching decoder/checkpoints and a
   labelled held-out set when those are available. Encoder output identity on
   development scans is useful evidence, not a complete accuracy evaluation.
4. Repeat the chosen candidate on the Jetson when moving to target deployment.
   Desktop latency and memory reductions do not determine Jetson results.

No additional experiments are scheduled or started automatically. The manual
entry point is `compare_topk_chunked.py`; `--chunk-sizes` selects the sizes to test.

## Limits before deployment decisions

Earlier methods used three development scans; the combined experiment used 20.
Bit-identical outputs refer only to the respective selected inputs. Relative latent L2 is a numerical diagnostic, **not accuracy**;
the previous 1% screening threshold is not an approved quality tolerance.
Classification, reconstruction and downstream RL quality were not measured in
these deployment experiments. Real-world labelled evaluation remains outstanding.

The latest TopK mean corresponds to about 5.28 scans/s; combined chunk size
1024 corresponds to about 5.16 scans/s, excluding preprocessing and transfer. It does not process every 10 Hz sensor scan. Whether every scan
must be encoded should be agreed with the supervisor; this is not automatically
a project failure. All figures are desktop results, not Jetson measurements.

The comparison scripts remain manual. The combined experiment above was run
only after the explicit user request. Reading this report starts no workload. Per-method reports in this folder retain checkpoint provenance,
input scan paths and individual findings.
