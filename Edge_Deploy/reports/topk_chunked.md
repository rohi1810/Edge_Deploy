# TopK plus chunking comparison

Recorded: 2026-10-27T13:00:45+00:00 (UTC).
Status: complete. GPU: Quadro RTX 5000.
PyTorch: 2.11.0+cu128; CUDA: 12.8.

20 scans from distinct scenarios; batch 1; 32768 points; FP32.
Two warmups and 5 timed forwards per scan, per variant.

| Variant | Mean ms | Median ms | p95 ms | Peak allocated MB | Speed vs original | Exact latent | Neighbor checks |
|---|---:|---:|---:|---:|---:|---|---|
| Original | 340.52 | 336.48 | 355.80 | 7125.79 | 1.000x | True | Reference |
| TopK | 189.49 | 186.56 | 211.13 | 1757.08 | 1.797x | True | 60/60 matched |
| TopK + chunk 128 | 209.53 | 203.80 | 239.17 | 81.69 | 1.625x | True | 60/60 matched |
| TopK + chunk 256 | 202.95 | 202.91 | 209.70 | 122.26 | 1.678x | True | 60/60 matched |
| TopK + chunk 512 | 208.73 | 210.66 | 232.37 | 231.24 | 1.631x | True | 60/60 matched |
| TopK + chunk 1024 | 193.69 | 191.97 | 209.69 | 449.22 | 1.758x | True | 60/60 matched |

## Result

Fastest numerically matching variant: **TopK**.
Fastest matching combination: **TopK + chunk 1024**.
Combination speed ratio vs TopK alone: 0.978x.
Combination allocated-memory reduction vs TopK alone: 74.43%.
Combination allocated-memory reduction vs original: 93.70%.

## Output checks

Exact latent compares the saved encoder output for every selected scan with the original.
Neighbor checks compare complete index tensors at all three encoder levels in separate untimed forwards.
Every timed forward also checks output shape and finiteness. Numerical equivalence on these inputs
does not establish classification accuracy, reconstruction quality or downstream RL performance.

## Measurement limits

Warm encoder forward only; excludes preprocessing, transfer, graph capture and correctness checks.
Memory is peak PyTorch allocated GPU memory, not total device/process memory.
FPS CUDA-graph caches are cleared between variants; both use the same existing sampling algorithm.
Variants run sequentially in table order. Clocks/power are not locked; timing is a development comparison.
Repeated forwards are not independent scenes. These are desktop results, not Jetson measurements.

Checkpoint: `C:\Users\rohi\PycharmProjects\sim2reallidar\weights\pointnet2_epoch39_encoder.pt`.
SHA256: `16b7845ef7f0db5a5062f3efbbf1c1722677424f0c17e6798f9721dd62bd5693`.

## Inputs

- `F:\training_data\o_robot_nav\00efe697-67a0-457e-9c32-059ddca089c5\2026_08_26_14_30_44_903.npz`
- `F:\training_data\o_robot_nav\046630c6-f6cc-44b3-825a-ebcfa3cc82c3\2026_08_26_14_44_36_807.npz`
- `F:\training_data\o_robot_nav\0dfd4381-6bcb-49bc-95f1-7f6cc68660de\2026_08_26_15_14_32_889.npz`
- `F:\training_data\o_robot_nav\122e86f0-8b3f-46f5-8eaa-9f6b1232c7df\2026_08_26_15_32_58_256.npz`
- `F:\training_data\o_robot_nav\1ebdbd60-b4d3-4fdf-99ce-86f76e8bb96c\2026_08_26_16_18_36_869.npz`
- `F:\training_data\o_robot_nav\2c97ac13-4f4a-4704-98df-2681024e0db3\2026_08_26_16_43_54_918.npz`
- `F:\training_data\o_robot_nav\3c361506-3caf-47df-972e-d950558e1f36\2026_08_26_17_42_39_446.npz`
- `F:\training_data\o_robot_nav\506049aa-ee0a-4f40-aeba-6739112312c3\2026_08_26_19_11_07_347.npz`
- `F:\training_data\o_robot_nav\68488671-e2ac-4a04-8570-59f45b26c582\2026_08_26_20_19_27_341.npz`
- `F:\training_data\o_robot_nav\77f470a0-84d9-4240-842b-2cf3125873f6\2026_08_26_21_33_07_059.npz`
- `F:\training_data\o_robot_nav\8cfaee08-fc69-4ba6-980e-f1f7d266d1af\2026_08_26_22_50_57_115.npz`
- `F:\training_data\o_robot_nav\9932a488-9b63-4f46-8d64-c3e868b854e9\2026_08_26_23_26_41_082.npz`
- `F:\training_data\o_robot_nav\ad641e67-fdf6-4e73-82d7-cd0744e1b46d\2026_08_27_00_03_49_946.npz`
- `F:\training_data\o_robot_nav\bcecf1c3-c58e-4443-b4f8-1ccb014ba2f5\2026_08_27_00_42_54_721.npz`
- `F:\training_data\o_robot_nav\c5255a0c-3f79-4953-a22a-665266256e35\2026_08_27_01_06_04_059.npz`
- `F:\training_data\o_robot_nav\cab97d42-bc62-4435-9774-1c16682d964c\2026_08_27_01_25_41_619.npz`
- `F:\training_data\o_robot_nav\d535c5f6-150d-44db-b7a5-dc6a45849c6c\2026_08_27_02_11_29_459.npz`
- `F:\training_data\o_robot_nav\d79402dc-4812-4987-8d30-e4920baf981c\2026_08_27_02_34_43_731.npz`
- `F:\training_data\o_robot_nav\eaf6f7d1-458b-43d3-8479-78fc57cb6e4e\2026_08_27_03_23_15_925.npz`
- `F:\training_data\o_robot_nav\f7e8f989-9046-4a88-86b9-ab39ce7dfe41\2026_08_27_03_45_23_210.npz`
