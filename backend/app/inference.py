"""
ONNX Runtime High-Performance Inference Engine for ArchiGen 3D.
Loads the optimized Generator ONNX model and provides low-latency inference.
Satisfies Section 5.1 & Section 6 of syllabus: 'backend/app/inference.py'
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import io
import onnxruntime as ort

class ArchiGenONNXInference:
    def __init__(self, model_path: str = "models/saved/generator.onnx"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            # Fallback path if running inside backend container
            alt_path = Path("../models/saved/generator.onnx")
            if alt_path.exists():
                self.model_path = alt_path
            else:
                raise FileNotFoundError(f"ONNX Model not found at: {self.model_path}")

        print(f"[INFERENCE ENGINE] Loading ONNX model from: {self.model_path}")
        # Initialize ONNX Runtime Inference Session
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[INFERENCE ENGINE] Loaded session with input: '{self.input_name}' -> output: '{self.output_name}'")

    def preprocess(self, image_bytes: bytes, target_size: int = 256) -> np.ndarray:
        """
        Preprocesses raw image bytes:
        - Resizes to (256, 256)
        - Normalizes to [-1.0, 1.0]
        - Formats to tensor shape (1, 3, 256, 256)
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
        
        # Convert to numpy array [0, 255] -> [-1, 1]
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        
        # Transpose from (H, W, C) to (C, H, W)
        arr = np.transpose(arr, (2, 0, 1))
        
        # Add batch dimension -> (1, 3, 256, 256)
        return np.expand_dims(arr, axis=0).astype(np.float32)

    def postprocess(self, output_tensor: np.ndarray) -> bytes:
        """
        Postprocesses output tensor from [-1.0, 1.0] back to PNG image bytes.
        """
        # Squeeze batch dimension -> (3, 256, 256)
        arr = output_tensor[0]
        
        # Transpose to (H, W, C)
        arr = np.transpose(arr, (1, 2, 0))
        
        # Denormalize from [-1, 1] to [0, 255]
        arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(arr)
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="PNG")
        return output_buffer.getvalue()

    def generate(self, image_bytes: bytes) -> bytes:
        """Runs end-to-end inference in < 300ms."""
        input_tensor = self.preprocess(image_bytes)
        output_tensor = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        return self.postprocess(output_tensor)
