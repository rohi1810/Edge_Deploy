# Edge deployment comparisons

Ahmed's code and weights are read-only inputs. Data stays on F:.

Start with [the combined comparison and next direction](reports/comparison.md).

One comparison script and one report are kept for each method:

| Script | Report |
|---|---|
| `compare_fp16.py` | `reports/fp16.md` |
| `compare_chunked.py` | `reports/chunked.md` |
| `compare_topk.py` | `reports/topk.md` |
| `compare_topk_chunked.py` | `reports/topk_chunked.md` |
| `compare_onnx.py` | `reports/onnx.md` |
| `compare_tensorrt.py` | `reports/tensorrt.md` |

`_support/` contains shared timing, export and runtime code. `.venv/` is the
required GPU Python environment, not an experiment cache.

## Run manually

From Ahmed's repository root in PowerShell:

```powershell
.\Edge_Deploy\.venv\Scripts\python.exe Edge_Deploy\compare_topk.py --recordings-dir F:\training_data\o_robot_nav
```

Replace `compare_topk.py` with the method you want. Optional arguments:
`--samples 3 --trials 3`. Chunked queries also accept `--chunk-size 128`.
TensorRT defaults to FP32; `--fp16` explicitly selects the previously rejected
experimental variant.

`compare_topk_chunked.py` compares original, TopK, and combined chunk sizes
128, 256, 512 and 1024. Use `--chunk-sizes` to select different sizes. It also
checks all neighborhood indices separately from the timed forwards.

Each command compares against the original encoder and replaces only its
method's Markdown report. ONNX exports are temporary, TensorRT engines stay
in memory, and no profiles, JSON reports, log files or bytecode caches are
saved by these scripts. Imports do not start experiments.

The reports state their measurement scope. The combined method was run on 20
scans after an explicit request; no experiments are started automatically.
Numerical output similarity does not establish classification or RL accuracy.
Desktop measurements are not Jetson results.

Use `.venv\Scripts\python.exe -m pip` for package management; standalone pip
launchers may retain the environment's previous location. `requirements.txt`
records the desktop dependencies. Jetson requires its JetPack-compatible stack.
