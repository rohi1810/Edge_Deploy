# Chunked neighborhood queries

Recorded on 2026-10-27, before the folder cleanup. No new run was performed.

Hardware: Quadro RTX 5000. Batch 1, 32768 points per scan.
3 recorded scans, 3 timed forwards per scan.

| Metric | Original FP32 | Candidate |
|---|---:|---:|
| Mean forward | 338.20 ms | 357.16 ms |
| Peak allocated GPU memory | 7125.79 MB | 235.53 MB |

Speed ratio: 0.947x.

Chunk size: 128. Neighborhood indices and final latents matched exactly on all selected scans.
Conclusion: 96.7% less peak allocated GPU memory, with 5.6% higher latency.

Warm encoder forward only; preprocessing, transfers, export and build are excluded.
These are desktop measurements. Classification accuracy and RL quality were not measured.

Checkpoint: `weights/pointnet2_epoch39_encoder.pt` in Ahmed's repository.
SHA256: `16b7845ef7f0db5a5062f3efbbf1c1722677424f0c17e6798f9721dd62bd5693`.

## Inputs

- `F:\training_data\o_robot_nav\00efe697-67a0-457e-9c32-059ddca089c5\2026_08_26_14_30_44_903.npz`
- `F:\training_data\o_robot_nav\7ed476fa-607b-477a-a0c0-40cc4d0cbab4\2026_08_26_22_09_54_607.npz`
- `F:\training_data\o_robot_nav\f7e8f989-9046-4a88-86b9-ab39ce7dfe41\2026_08_27_03_45_23_210.npz`
