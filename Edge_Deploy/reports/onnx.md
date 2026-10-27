# ONNX Runtime CUDA

Recorded on 2026-10-27, before the folder cleanup. No new run was performed.

Hardware: Quadro RTX 5000. Batch 1, 32768 points per scan.
3 recorded scans, 5 timed forwards per scan.

| Metric | Original FP32 | Candidate |
|---|---:|---:|
| Mean forward | 339.69 ms | 1392.04 ms |
| Peak allocated GPU memory | 7125.79 MB | Not measured |

Speed ratio: 0.244x.

ONNX 1.19.1; ONNX Runtime GPU 1.23.2; PyTorch 2.11.0+cu128.
Complete encoder export uses TopK queries and three compact FPS loops; fixed input shape (1,32768,3).
Prior profiling recorded 48608 CUDA node executions and 26 CPU node executions.
Conclusion: numerically close outputs, but slower than PyTorch.
The 1% relative-L2 screening threshold is an engineering diagnostic, not a task-accuracy guarantee.

Relative latent L2 differences: 0.002240%, 0.001203%, 0.000054%.

Warm encoder forward only; preprocessing, transfers, export and build are excluded.
These are desktop measurements. Classification accuracy and RL quality were not measured.

Checkpoint: `weights/pointnet2_epoch39_encoder.pt` in Ahmed's repository.
SHA256: `16b7845ef7f0db5a5062f3efbbf1c1722677424f0c17e6798f9721dd62bd5693`.

## Inputs

- `F:\training_data\o_robot_nav\00efe697-67a0-457e-9c32-059ddca089c5\2026_08_26_14_30_44_903.npz`
- `F:\training_data\o_robot_nav\7ed476fa-607b-477a-a0c0-40cc4d0cbab4\2026_08_26_22_09_54_607.npz`
- `F:\training_data\o_robot_nav\f7e8f989-9046-4a88-86b9-ab39ce7dfe41\2026_08_27_03_45_23_210.npz`
