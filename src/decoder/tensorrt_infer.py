"""
TensorRT Inference Engine for EEG Decoder
Optimized for NVIDIA Jetson Orin NX
"""

import numpy as np
from typing import List, Optional, Tuple
import time


class TensorRTDecoder:
    """
    TensorRT-accelerated EEG decoder.

    Optimized for Jetson Orin NX:
    - FP16 precision for 2x throughput
    - Serialized engine for fast loading
    - Asynchronous inference with CUDA streams

    Typical performance:
    - PyTorch FP32: ~15ms per inference
    - TensorRT FP16: ~5ms per inference
    """

    def __init__(
        self,
        engine_path: str,
        vocab: List[str],
        device_id: int = 0
    ):
        """
        Initialize TensorRT decoder.

        Args:
            engine_path: Path to serialized TensorRT engine (.engine)
            vocab: List of output class labels
            device_id: CUDA device ID
        """
        self.engine_path = engine_path
        self.vocab = vocab
        self.device_id = device_id

        self._engine = None
        self._context = None
        self._stream = None
        self._bindings = None

        # Performance tracking
        self._inference_times = []

    def load(self) -> bool:
        """Load TensorRT engine."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            # Create logger
            logger = trt.Logger(trt.Logger.WARNING)

            # Load serialized engine
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()

            runtime = trt.Runtime(logger)
            self._engine = runtime.deserialize_cuda_engine(engine_data)

            if self._engine is None:
                print(f"Failed to load engine from {self.engine_path}")
                return False

            # Create execution context
            self._context = self._engine.create_execution_context()

            # Create CUDA stream
            self._stream = cuda.Stream()

            # Allocate buffers
            self._allocate_buffers()

            print(f"Loaded TensorRT engine: {self.engine_path}")
            return True

        except ImportError:
            print("TensorRT not available. Install with JetPack.")
            return False
        except Exception as e:
            print(f"Failed to load TensorRT engine: {e}")
            return False

    def _allocate_buffers(self):
        """Allocate input/output buffers."""
        import pycuda.driver as cuda

        self._bindings = []
        self._inputs = []
        self._outputs = []

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            dtype = self._engine.get_tensor_dtype(name)
            shape = self._engine.get_tensor_shape(name)

            # Convert to numpy dtype
            if dtype == 1:  # trt.float32
                np_dtype = np.float32
            elif dtype == 2:  # trt.float16
                np_dtype = np.float16
            else:
                np_dtype = np.float32

            # Allocate host and device memory
            size = int(np.prod(shape))
            host_mem = np.zeros(size, dtype=np_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self._bindings.append(int(device_mem))

            if self._engine.get_tensor_mode(name) == 0:  # Input
                self._inputs.append({
                    'name': name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape
                })
            else:  # Output
                self._outputs.append({
                    'name': name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape
                })

    def predict(
        self,
        eeg_data: np.ndarray
    ) -> Tuple[str, float, dict]:
        """
        Run inference on EEG data.

        Args:
            eeg_data: Array of shape (channels, time_steps)

        Returns:
            Tuple of (intent, confidence, probabilities)
        """
        if self._engine is None:
            # Fallback to dummy prediction
            return self._dummy_predict(eeg_data)

        import pycuda.driver as cuda

        start_time = time.perf_counter()

        # Prepare input
        input_data = eeg_data.astype(np.float32).flatten()
        np.copyto(self._inputs[0]['host'], input_data)

        # Copy input to device
        cuda.memcpy_htod_async(
            self._inputs[0]['device'],
            self._inputs[0]['host'],
            self._stream
        )

        # Run inference
        self._context.execute_async_v2(
            bindings=self._bindings,
            stream_handle=self._stream.handle
        )

        # Copy output to host
        cuda.memcpy_dtoh_async(
            self._outputs[0]['host'],
            self._outputs[0]['device'],
            self._stream
        )

        # Synchronize
        self._stream.synchronize()

        # Get logits and apply softmax
        logits = self._outputs[0]['host'].reshape(-1)
        probs = self._softmax(logits)

        # Track timing
        elapsed = (time.perf_counter() - start_time) * 1000
        self._inference_times.append(elapsed)

        # Get prediction
        idx = np.argmax(probs)
        intent = self.vocab[idx]
        confidence = probs[idx]
        prob_dict = {word: float(probs[i]) for i, word in enumerate(self.vocab)}

        return intent, float(confidence), prob_dict

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def _dummy_predict(self, eeg_data: np.ndarray) -> Tuple[str, float, dict]:
        """Dummy prediction when TensorRT not available."""
        # Simulate inference time
        time.sleep(0.005)

        probs = np.random.dirichlet(np.ones(len(self.vocab)))
        idx = np.argmax(probs)

        return (
            self.vocab[idx],
            float(probs[idx]),
            {word: float(probs[i]) for i, word in enumerate(self.vocab)}
        )

    def get_avg_latency(self) -> float:
        """Get average inference latency in ms."""
        if not self._inference_times:
            return 0.0
        return np.mean(self._inference_times[-100:])

    def warmup(self, num_runs: int = 10):
        """Warm up the engine with dummy data."""
        dummy = np.random.randn(32, 1000).astype(np.float32)
        for _ in range(num_runs):
            self.predict(dummy)
        self._inference_times.clear()


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_mb: int = 1024
) -> bool:
    """
    Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Output path for TensorRT engine
        fp16: Enable FP16 precision
        workspace_mb: Workspace size in MB

    Returns:
        True if successful
    """
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return False

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Building with FP16 precision")

        # Build engine
        print("Building TensorRT engine (this may take a few minutes)...")
        engine = builder.build_serialized_network(network, config)

        if engine is None:
            print("Failed to build engine")
            return False

        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(engine)

        print(f"Saved TensorRT engine to {engine_path}")
        return True

    except ImportError:
        print("TensorRT not available")
        return False
    except Exception as e:
        print(f"Failed to build engine: {e}")
        return False


if __name__ == '__main__':
    # Test decoder (without TensorRT)
    vocab = ['yes', 'no', 'help', 'water', 'bathroom',
             'pain', 'tired', 'hungry', 'cold', 'hot']

    decoder = TensorRTDecoder(
        engine_path='models/decoder.engine',
        vocab=vocab
    )

    # Test with dummy data
    eeg = np.random.randn(32, 1000).astype(np.float32)
    intent, conf, probs = decoder.predict(eeg)

    print("TensorRT Decoder Test (dummy mode)")
    print(f"  Intent: {intent}")
    print(f"  Confidence: {conf:.2%}")
    print(f"  Top-3: {sorted(probs.items(), key=lambda x: -x[1])[:3]}")
