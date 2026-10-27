"""Compare original, TopK, and chunked TopK. Runs only when explicitly invoked."""
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
from pathlib import Path
import sys

sys.dont_write_bytecode = True
EDGE = Path(__file__).resolve().parent
ROOT = EDGE.parent
sys.path.insert(0, str(ROOT))

import torch
from dataset import _load_raw
from models.pointnet2 import PointNet2Encoder
from models import pointnet2_utils as ptu
from _support.common import benchmark, select_scans
from _support.onnx_export import chunked_topk_query, topk_query


def check_neighbors(model, inputs, candidate_query, original_query):
    """Untimed full-encoder forwards check every neighborhood at all levels."""
    checks = mismatches = 0

    def checked_query(radius, k, xyz, centroids):
        nonlocal checks, mismatches
        expected = original_query(radius, k, xyz, centroids)
        actual = candidate_query(radius, k, xyz, centroids)
        checks += 1
        mismatches += int(not torch.equal(expected, actual))
        return actual

    previous = ptu.ball_query
    try:
        ptu.ball_query = checked_query
        with torch.inference_mode():
            for value in inputs:
                model(value.cuda())
    finally:
        ptu.ball_query = previous
    return checks, mismatches


def write_report(args, scans, rows, complete):
    baseline, topk = rows[0], next((row for row in rows if row['name'] == 'TopK'), None)
    lines = [
        '# TopK plus chunking comparison', '',
        f"Recorded: {datetime.now(timezone.utc).isoformat(timespec='seconds')} (UTC).",
        f"Status: {'complete' if complete else 'in progress'}. GPU: {torch.cuda.get_device_name(0)}.",
        f'PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}.', '',
        f'{len(scans)} scans from distinct scenarios; batch 1; 32768 points; FP32.',
        f'Two warmups and {args.trials} timed forwards per scan, per variant.', '',
        '| Variant | Mean ms | Median ms | p95 ms | Peak allocated MB | Speed vs original | Exact latent | Neighbor checks |',
        '|---|---:|---:|---:|---:|---:|---|---|',
    ]
    for row in rows:
        metrics = row['metrics']
        neighbors = ('Reference' if row['checks'] == 0 else
                     f"{row['checks'] - row['mismatches']}/{row['checks']} matched")
        lines.append(
            f"| {row['name']} | {metrics['latency_ms_mean']:.2f} | "
            f"{metrics['latency_ms_median']:.2f} | {metrics['latency_ms_p95']:.2f} | "
            f"{metrics['peak_gpu_allocated_mb']:.2f} | "
            f"{baseline['metrics']['latency_ms_mean'] / metrics['latency_ms_mean']:.3f}x | "
            f"{row['exact']} | {neighbors} |")
    valid = [row for row in rows[1:] if row['exact'] and row['mismatches'] == 0]
    combined = [row for row in valid if row['chunk_size'] is not None]
    if complete and combined and topk:
        best = min(combined, key=lambda row: row['metrics']['latency_ms_mean'])
        metrics, topk_metrics = best['metrics'], topk['metrics']
        fastest = min(valid, key=lambda row: row['metrics']['latency_ms_mean'])
        lines += [
            '', '## Result', '',
            f"Fastest numerically matching variant: **{fastest['name']}**.",
            f"Fastest matching combination: **{best['name']}**.",
            f"Combination speed ratio vs TopK alone: {topk_metrics['latency_ms_mean'] / metrics['latency_ms_mean']:.3f}x.",
            f"Combination allocated-memory reduction vs TopK alone: "
            f"{1 - metrics['peak_gpu_allocated_mb'] / topk_metrics['peak_gpu_allocated_mb']:.2%}.",
            f"Combination allocated-memory reduction vs original: "
            f"{1 - metrics['peak_gpu_allocated_mb'] / baseline['metrics']['peak_gpu_allocated_mb']:.2%}.",
        ]
    lines += [
        '', '## Output checks', '',
        'Exact latent compares the saved encoder output for every selected scan with the original.',
        'Neighbor checks compare complete index tensors at all three encoder levels in separate untimed forwards.',
        'Every timed forward also checks output shape and finiteness. Numerical equivalence on these inputs',
        'does not establish classification accuracy, reconstruction quality or downstream RL performance.', '',
        '## Measurement limits', '',
        'Warm encoder forward only; excludes preprocessing, transfer, graph capture and correctness checks.',
        'Memory is peak PyTorch allocated GPU memory, not total device/process memory.',
        'FPS CUDA-graph caches are cleared between variants; both use the same existing sampling algorithm.',
        'Variants run sequentially in table order. Clocks/power are not locked; timing is a development comparison.',
        'Repeated forwards are not independent scenes. These are desktop results, not Jetson measurements.', '',
        f'Checkpoint: `{args.checkpoint.resolve()}`.',
        f'SHA256: `{hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}`.', '',
        '## Inputs', '', *[f'- `{path}`' for path in scans], '',
    ]
    destination = EDGE / 'reports/topk_chunked.md'
    destination.write_text('\n'.join(lines), encoding='utf-8')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT / 'weights/pointnet2_epoch39_encoder.pt')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--chunk-sizes', type=int, nargs='+', default=[128, 256, 512, 1024])
    args = parser.parse_args()
    if min([args.samples, args.trials, *args.chunk_sizes]) < 1:
        parser.error('samples, trials and chunk sizes must be positive')
    if not torch.cuda.is_available():
        parser.error('CUDA-enabled PyTorch and an NVIDIA GPU are required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scans = select_scans(args.recordings_dir, args.samples)
    inputs = [torch.from_numpy(_load_raw(path)[0]).unsqueeze(0) for path in scans]
    model = PointNet2Encoder().cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    original_query = ptu.ball_query
    variants = [('Original', None, original_query), ('TopK', None, topk_query)]
    variants += [(f'TopK + chunk {size}', size, partial(chunked_topk_query, chunk_size=size))
                 for size in dict.fromkeys(args.chunk_sizes)]
    rows, reference = [], None
    try:
        for name, size, query in variants:
            print(f'\n{name}:', flush=True)
            ptu.ball_query = query
            metrics, output = benchmark(model, inputs, 'fp32', torch.device('cuda'), 2, args.trials)
            if reference is None:
                reference = output
                checks = mismatches = 0
            else:
                print('Checking all neighborhood indices (untimed)...', flush=True)
                checks, mismatches = check_neighbors(model, inputs, query, original_query)
            row = {'name': name, 'chunk_size': size, 'metrics': metrics,
                   'exact': torch.equal(output, reference), 'checks': checks, 'mismatches': mismatches}
            rows.append(row)
            print(f"Mean {metrics['latency_ms_mean']:.2f} ms; "
                  f"peak {metrics['peak_gpu_allocated_mb']:.2f} MB; "
                  f"exact latent: {row['exact']}; neighbor mismatches: {mismatches}", flush=True)
            report = write_report(args, scans, rows, complete=len(rows) == len(variants))
        print(f'\nReport: {report}', flush=True)
    finally:
        ptu.ball_query = original_query
        ptu._fps_graph_cache.clear()


if __name__ == '__main__':
    main()
