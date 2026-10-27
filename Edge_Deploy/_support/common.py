"""Shared scan loading, encoder timing, and precision helpers."""
from contextlib import nullcontext
import gc
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from models import pointnet2_utils as ptu

def encode(model, xyz, precision):
    """Keep the coordinate tensor FP32 in both precision modes."""
    context = (torch.autocast(device_type="cuda", dtype=torch.float16)
               if precision == "fp16" else nullcontext())
    with context:
        latent, _ = model(xyz)
    return latent


def select_scans(root, count):
    # Small representative selection spread over scenarios, not the first
    # consecutive frames. These are benchmark inputs, never a held-out test set.
    scenarios = sorted(p for p in root.iterdir() if p.is_dir())
    if not scenarios:
        scenarios = [root]
    candidates = []
    for scenario in scenarios:
        scans = sorted(scenario.glob("*.npz"))
        if scans:
            candidates.append(scans[len(scans) // 2])
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} scenarios with scans; reduce --samples")
    indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[i] for i in indices]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(model, inputs, precision, device, warmup, trials):
    synchronize(device)
    ptu._fps_graph_cache.clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    outputs, timings = [], []
    peak_allocated = peak_reserved = 0
    for index, cpu_input in enumerate(inputs):
        xyz = cpu_input.to(device)
        with torch.inference_mode():
            # At least one untimed call: captures FPS graphs and initializes
            # libraries. Cold startup is deliberately excluded from this result.
            for _ in range(max(1, warmup)):
                latent = encode(model, xyz, precision)
            del latent
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            sample_timings = []
            for _ in range(trials):
                synchronize(device)
                start = time.perf_counter()
                latent = encode(model, xyz, precision)
                synchronize(device)
                sample_timings.append((time.perf_counter() - start) * 1000)
                # Checks and output transfer are outside the timing region.
                if latent.shape != (1, 256) or not torch.isfinite(latent).all():
                    raise RuntimeError(f"Invalid {precision} latent for scan {index}")
                saved = latent.float().cpu().clone()
                del latent
            outputs.append(saved)
            timings.extend(sample_timings)
            if device.type == "cuda":
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
        del xyz
        print(f"  {precision}: scan {index + 1}/{len(inputs)}, "
              f"median {np.median(sample_timings):.1f} ms", flush=True)
    result = {
        "precision": precision,
        "latency_ms_mean": float(np.mean(timings)),
        "latency_ms_median": float(np.median(timings)),
        "latency_ms_p95": float(np.percentile(timings, 95)),
        "measurements": len(timings),
        "latencies_ms": timings,
        "peak_gpu_allocated_mb": peak_allocated / 1e6 if device.type == "cuda" else None,
        "peak_gpu_reserved_mb": peak_reserved / 1e6 if device.type == "cuda" else None,
        "fps_cuda_graphs_active": bool(ptu._fps_graph_cache),
        "output_dtype": "float16" if precision == "fp16" else "float32",
    }
    return result, torch.cat(outputs)


