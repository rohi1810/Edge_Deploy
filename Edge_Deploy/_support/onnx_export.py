"""Static batch-one ONNX encoder with compact FPS loops and TopK queries.

Ahmed's modules and checkpoint remain untouched. Export the neural/query graph
with centroid indices as inputs, then connect standard ONNX Loop FPS nodes.
This avoids tracing 5376 Python iterations into hundreds of thousands of nodes.
"""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from models.pointnet2_utils import index_points


def topk_query(radius, k, xyz, centroids):
    distances = torch.cdist(centroids, xyz, compute_mode='use_mm_for_euclid_dist') ** 2
    nearest = distances.argmin(dim=-1)
    n = xyz.shape[1]
    grid = torch.arange(n, device=xyz.device).view(1, 1, n)
    sentinel = torch.tensor(n, dtype=torch.long, device=xyz.device)
    candidates = torch.where(distances <= radius ** 2, grid, sentinel)
    values = torch.topk(candidates, min(k, n), dim=-1, largest=False, sorted=True).values
    values = torch.where(values == n, nearest.unsqueeze(-1).expand_as(values), values)
    if n < k:
        values = torch.cat([values, nearest.unsqueeze(-1).expand(-1, -1, k - n)], dim=-1)
    return values


def chunked_topk_query(radius, k, xyz, centroids, chunk_size=128):
    """Select the same neighbors with smaller intermediate tensors."""
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    return torch.cat([
        topk_query(radius, k, xyz, centroids[:, start:start + chunk_size])
        for start in range(0, centroids.shape[1], chunk_size)
    ], dim=1)


class IndexedEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, xyz, idx1, idx2, idx3):
        points = None
        coordinates = []
        for stage, indices in zip(
                [self.encoder.sa1, self.encoder.sa2, self.encoder.sa3], [idx1, idx2, idx3]):
            centroids = index_points(xyz, indices)
            groups = topk_query(stage.radius, stage.k, xyz, centroids)
            relative = index_points(xyz, groups) - centroids.unsqueeze(2)
            grouped = (relative if points is None else
                       torch.cat([relative, index_points(points, groups)], dim=-1))
            points = stage.mlp(grouped.permute(0, 3, 2, 1)).max(dim=2).values.permute(0, 2, 1)
            xyz = centroids
            coordinates.append(xyz)
        latent = self.encoder.head(points.max(dim=1).values)
        return latent, coordinates[0], coordinates[1]


def fps_nodes(prefix, xyz, indices_name, n, samples):
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as T

    def name(value):
        return f'{prefix}_{value}'

    def tensor(value, dtype, dims):
        return h.make_tensor_value_info(name(value), dtype, dims)

    initializers = [
        nh.from_array(np.array([0], np.int64), name('axis0')),
        nh.from_array(np.array([1], np.int64), name('axis1')),
        nh.from_array(np.full(n, 1e10, np.float32), name('distance0')),
        nh.from_array(np.array(0, np.int64), name('farthest0')),
        nh.from_array(np.array(samples, np.int64), name('trip')),
    ]
    body = h.make_graph([
        h.make_node('Gather', [name('points'), name('farthest')], [name('centroid')], axis=0),
        h.make_node('Sub', [name('points'), name('centroid')], [name('diff')]),
        h.make_node('Mul', [name('diff'), name('diff')], [name('square')]),
        h.make_node('ReduceSum', [name('square'), name('axis1')], [name('dist')], keepdims=0),
        h.make_node('Min', [name('distance'), name('dist')], [name('updated')]),
        h.make_node('Unsqueeze', [name('updated'), name('axis0')], [name('updated_2d')]),
        h.make_node('ArgMax', [name('updated_2d')], [name('next_vector')], axis=1, keepdims=0, select_last_index=0),
        h.make_node('Squeeze', [name('next_vector'), name('axis0')], [name('next')]),
        h.make_node('Constant', [], [name('condition_out')],
                    value=h.make_tensor(name('always_true'), T.BOOL, [], [True])),
        h.make_node('Identity', [name('farthest')], [name('selected')]),
    ], name('body'),
        [tensor('iteration', T.INT64, []), tensor('condition', T.BOOL, []),
         tensor('distance', T.FLOAT, [n]), tensor('farthest', T.INT64, [])],
        [tensor('condition_out', T.BOOL, []), tensor('updated', T.FLOAT, [n]),
         tensor('next', T.INT64, []), tensor('selected', T.INT64, [])])
    nodes = [
        h.make_node('Squeeze', [xyz, name('axis0')], [name('points')], name=name('squeeze')),
        h.make_node('Loop', [name('trip'), '', name('distance0'), name('farthest0')],
                    [name('distance_final'), name('farthest_final'), name('indices_flat')],
                    body=body, name=name('loop')),
        h.make_node('Unsqueeze', [name('indices_flat'), name('axis0')], [indices_name], name=name('indices')),
    ]
    # Loop bodies capture points/axis1 from the enclosing graph.
    return nodes, initializers, {name('loop'): {name('points'), name('axis1')}}


def export_encoder(encoder, destination, point_count=32768):
    import onnx
    from onnx import helper as h

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    device = next(encoder.parameters()).device
    counts = [encoder.sa1.n_samples, encoder.sa2.n_samples, encoder.sa3.n_samples]
    x = torch.zeros(1, point_count, 3, device=device)
    ids = [torch.zeros(1, count, dtype=torch.long, device=device) for count in counts]
    with torch.inference_mode():
        torch.onnx.export(
            IndexedEncoder(encoder).eval(), (x, *ids), str(destination),
            input_names=['points', 'fps1_indices', 'fps2_indices', 'fps3_indices'],
            output_names=['latent', 'stage1_xyz', 'stage2_xyz'],
            opset_version=17, dynamo=False, do_constant_folding=True)
    model = onnx.load(str(destination))
    nodes = list(model.graph.node)
    initializers = list(model.graph.initializer)
    dependencies = {}
    for level, xyz, n, count in zip(
            [1, 2, 3], ['points', 'stage1_xyz', 'stage2_xyz'],
            [point_count, *counts[:2]], counts):
        extra, constants, captured = fps_nodes(f'fps{level}', xyz, f'fps{level}_indices', n, count)
        nodes.extend(extra)
        initializers.extend(constants)
        dependencies.update(captured)
    # Topological sort, including values captured by the Loop bodies.
    ready = {value.name for value in initializers} | {'points'}
    ordered = []
    while nodes:
        waiting = []
        for node in nodes:
            required = set(node.input) - {''}
            required.update(dependencies.get(node.name, set()))
            if required <= ready:
                ordered.append(node)
                ready.update(node.output)
            else:
                waiting.append(node)
        if len(waiting) == len(nodes):
            raise RuntimeError(f'Unresolved export dependencies: {[n.name for n in waiting[:5]]}')
        nodes = waiting
    graph = h.make_graph(ordered, 'PointNet2Encoder', [model.graph.input[0]],
                         [model.graph.output[0]], initializer=initializers)
    model = h.make_model(graph, producer_name='Edge_Deploy', opset_imports=[h.make_opsetid('', 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(destination))
    return {'path': str(destination), 'nodes': len(ordered), 'fps_loops': 3,
            'input_shape': [1, point_count, 3], 'output_shape': [1, 256],
            'size_bytes': destination.stat().st_size}
