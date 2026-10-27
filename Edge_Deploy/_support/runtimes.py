"""ONNX/TensorRT GPU adapters. No profiles or engine files are saved."""
from pathlib import Path
import time

import numpy as np
import torch


class OnnxRunner:
    def __init__(self, model_path):
        import onnxruntime as ort
        ort.preload_dlls()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options,
            providers=[('CUDAExecutionProvider', {
                'user_compute_stream': str(torch.cuda.current_stream().cuda_stream),
                'do_copy_in_default_stream': True, 'use_tf32': False,
            })])
        if 'CUDAExecutionProvider' not in self.session.get_providers():
            raise RuntimeError('ONNX CUDA provider did not load; CPU fallback rejected')
        self.version = ort.__version__

    def __call__(self, xyz):
        xyz = xyz.contiguous()
        output = torch.empty(1, 256, device='cuda', dtype=torch.float32)
        binding = self.session.io_binding()
        binding.bind_input('points', 'cuda', 0, np.float32, tuple(xyz.shape), xyz.data_ptr())
        binding.bind_output('latent', 'cuda', 0, np.float32, (1, 256), output.data_ptr())
        self.session.run_with_iobinding(binding)
        return output

class TensorRTRunner:
    def __init__(self, model_path, fp16=False, workspace_mb=4096):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        if not parser.parse_from_file(str(Path(model_path).resolve())):
            raise RuntimeError('TensorRT ONNX parse failed:\n' + '\n'.join(
                str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.builder_optimization_level = 3
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            # Sampling/geometry decisions must stay FP32. Only convolutional
            # layers are allowed to use FP16 in this first TensorRT experiment.
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            arithmetic = {trt.LayerType.ELEMENTWISE, trt.LayerType.REDUCE,
                          trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.UNARY,
                          trt.LayerType.NORMALIZATION, trt.LayerType.CONVOLUTION}
            for layer in network:
                if layer.type in arithmetic:
                    layer.precision = (trt.float16 if layer.type == trt.LayerType.CONVOLUTION
                                       else trt.float32)
                # Compute precision alone does not constrain output storage.
                # Prevent geometric tensors from being silently rounded to half
                # between FP32 operations, including inside the sampling loops.
                for i in range(layer.num_outputs):
                    if layer.get_output(i).dtype == trt.float32:
                        layer.set_output_type(i, trt.float32)
        started = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        self.build_seconds = time.perf_counter() - started
        if serialized is None:
            raise RuntimeError('TensorRT engine build failed; see captured build log')
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError('TensorRT engine could not be deserialized')
        self.context = self.engine.create_execution_context()
        self.version = trt.__version__
        self.dtype_map = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.output_shape = tuple(self.engine.get_tensor_shape('latent'))
        self.output_dtype = self.dtype_map[self.engine.get_tensor_dtype('latent')]
        self.stream = torch.cuda.Stream()

    def __call__(self, xyz):
        xyz = xyz.contiguous()
        output = torch.empty(self.output_shape, device='cuda', dtype=self.output_dtype)
        self.context.set_tensor_address('points', xyz.data_ptr())
        self.context.set_tensor_address('latent', output.data_ptr())
        current = torch.cuda.current_stream()
        self.stream.wait_stream(current)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('TensorRT execute_async_v3 failed')
        current.wait_stream(self.stream)
        xyz.record_stream(self.stream)
        output.record_stream(self.stream)
        return output
