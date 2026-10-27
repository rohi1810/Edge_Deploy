"""Manual comparisons. Each run replaces only its method's Markdown report."""
import argparse
import gc
import hashlib
from pathlib import Path
import sys
import tempfile
import time

sys.dont_write_bytecode = True
EDGE = Path(__file__).resolve().parents[1]
ROOT = EDGE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from dataset import _load_raw
from models.pointnet2 import PointNet2Encoder
from models import pointnet2_utils as ptu
from .common import benchmark, select_scans

_ORIGINAL_QUERY = ptu.ball_query


def chunked_query(radius, k, xyz, centroids, chunk_size=128):
    if xyz.shape[1] <= 25:
        return _ORIGINAL_QUERY(radius, k, xyz, centroids)
    return torch.cat([
        _ORIGINAL_QUERY(radius, k, xyz, centroids[:, start:start + chunk_size])
        for start in range(0, centroids.shape[1], chunk_size)
    ], dim=1)


def measure_runtime(runner, inputs, trials):
    outputs, timings = [], []
    with torch.inference_mode():
        for value in inputs:
            xyz = value.cuda()
            warmup = runner(xyz)
            torch.cuda.synchronize()
            del warmup
            for _ in range(trials):
                torch.cuda.synchronize()
                started = time.perf_counter()
                latent = runner(xyz)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - started) * 1000)
                if latent.shape != (1, 256) or not torch.isfinite(latent).all():
                    raise RuntimeError('Invalid runtime latent')
                saved = latent.float().cpu().clone()
                del latent
            outputs.append(saved)
    return {
        'latency_ms_mean': float(np.mean(timings)),
        'peak_gpu_allocated_mb': None,
    }, torch.cat(outputs)


def write_report(method, args, scans, baseline, candidate, reference, output, details):
    difference = output - reference
    relative = difference.norm(dim=1) / reference.norm(dim=1).clamp_min(1e-12)
    memory = candidate['peak_gpu_allocated_mb']
    lines = [
        f'# {method.upper()} comparison', '',
        f'GPU: {torch.cuda.get_device_name(0)}; PyTorch: {torch.__version__}.', '',
        f'{len(scans)} scans; batch 1; 32768 points; {args.trials} timed forwards per scan.', '',
        '| Metric | Original FP32 | Candidate |', '|---|---:|---:|',
        f"| Mean forward | {baseline['latency_ms_mean']:.2f} ms | {candidate['latency_ms_mean']:.2f} ms |",
        f"| Peak allocated GPU memory | {baseline['peak_gpu_allocated_mb']:.2f} MB | "
        + (f'{memory:.2f} MB |' if memory is not None else 'Not measured |'), '',
        f"Speed ratio: {baseline['latency_ms_mean'] / candidate['latency_ms_mean']:.3f}x.",
        f'Bit-identical outputs on selected scans: {torch.equal(output, reference)}.',
        f'Maximum relative latent L2 difference: {relative.max().item():.6%}.',
        f'Maximum absolute latent difference: {difference.abs().max().item():.8f}.', '',
        'Numerical screen: ' + ('passed' if bool((relative <= 0.01).all()) else 'REJECTED')
        + ' (relative latent L2 <= 1% per scan; not an accuracy guarantee).', '',
        'Warm encoder forward only; excludes preprocessing, transfers, export and build.',
        'Desktop results only; classification and RL quality are not evaluated.', '',
        f'Checkpoint: `{args.checkpoint.resolve()}`.',
        f'SHA256: `{hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}`.', '',
        *details, '', '## Inputs', '', *[f'- `{scan}`' for scan in scans], '',
    ]
    destination = EDGE / 'reports' / f'{method}.md'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Report: {destination}')


def run(method):
    parser = argparse.ArgumentParser(description=f'Compare {method} with the original encoder.')
    parser.add_argument('--recordings-dir', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT / 'weights/pointnet2_epoch39_encoder.pt')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--trials', type=int, default=3)
    if method == 'chunked':
        parser.add_argument('--chunk-size', type=int, default=128)
    if method == 'tensorrt':
        parser.add_argument('--fp16', action='store_true',
                            help='Experimental; previously rejected for large output differences')
    args = parser.parse_args()
    if min(args.samples, args.trials) < 1 or getattr(args, 'chunk_size', 1) < 1:
        parser.error('samples, trials and chunk-size must be positive')
    if not torch.cuda.is_available():
        parser.error('CUDA-enabled PyTorch and an NVIDIA GPU are required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scans = select_scans(args.recordings_dir, args.samples)
    inputs = [torch.from_numpy(_load_raw(path)[0]).unsqueeze(0) for path in scans]
    model = PointNet2Encoder().cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    print('Original FP32 encoder:')
    baseline, reference = benchmark(model, inputs, 'fp32', torch.device('cuda'), 2, args.trials)
    details = []
    original_query = ptu.ball_query
    try:
        print(f'Candidate: {method}')
        if method in ('fp16', 'chunked', 'topk'):
            precision = 'fp16' if method == 'fp16' else 'fp32'
            if method == 'chunked':
                ptu.ball_query = lambda radius, k, xyz, centroids: chunked_query(
                    radius, k, xyz, centroids, args.chunk_size)
                details.append(f'Query chunk size: {args.chunk_size}.')
            elif method == 'topk':
                from .onnx_export import topk_query
                ptu.ball_query = topk_query
            candidate, output = benchmark(model, inputs, precision, torch.device('cuda'), 2, args.trials)
        else:
            from .onnx_export import export_encoder
            from .runtimes import OnnxRunner, TensorRTRunner

            ptu._fps_graph_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()
            with tempfile.TemporaryDirectory(prefix='.run_', dir=EDGE) as temporary:
                model_path = Path(temporary) / 'encoder.onnx'
                export_encoder(model, model_path)
                runner = (OnnxRunner(model_path) if method == 'onnx' else
                          TensorRTRunner(model_path, fp16=args.fp16))
                candidate, output = measure_runtime(runner, inputs, args.trials)
                details.append(f'Runtime version: {runner.version}.')
                details.append('Export uses TopK queries and compact FPS loops.')
                if method == 'onnx':
                    details.append(f'Enabled providers: {runner.session.get_providers()}; CPU/CUDA execution may mix.')
                else:
                    details.append(f'TensorRT convolution FP16 enabled: {args.fp16}.')
                    details.append(f'Build time, excluded from timing: {runner.build_seconds:.1f} s.')
                del runner
                gc.collect()
        write_report(method, args, scans, baseline, candidate, reference, output, details)
    finally:
        ptu.ball_query = original_query
        ptu._fps_graph_cache.clear()
