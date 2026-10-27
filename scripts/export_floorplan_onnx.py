import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import onnxruntime as ort
from src.floorplan.models import MultiTaskFloorplanUNet

class ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["struct"], out["room"]

def export_and_verify(
    checkpoint_path="models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth",
    onnx_path="models/saved/floorplan_unetpp/floorplan_multitask.onnx"
):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    device = torch.device("cpu")

    print(f"Loading checkpoint: {checkpoint_path}")
    model = MultiTaskFloorplanUNet(encoder_weights=None)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    wrapper = ONNXWrapper(model)
    dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float32)

    with torch.no_grad():
        torch_struct, torch_room = wrapper(dummy_input)

    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["struct_logits", "room_logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "struct_logits": {0: "batch_size"},
            "room_logits": {0: "batch_size"}
        }
    )

    print("Verifying ONNX Parity with ONNXRuntime...")
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input": dummy_input.numpy()}
    ort_struct, ort_room = ort_session.run(None, ort_inputs)

    diff_struct = np.max(np.abs(torch_struct.numpy() - ort_struct))
    diff_room = np.max(np.abs(torch_room.numpy() - ort_room))

    print(f"Max struct difference: {diff_struct:.6e}")
    print(f"Max room difference:   {diff_room:.6e}")
    assert diff_struct < 1e-4, f"Struct parity failed: {diff_struct}"
    assert diff_room < 1e-4, f"Room parity failed: {diff_room}"
    print(f"[SUCCESS] ONNX export {onnx_path} verified with sub-1e-4 parity!")

if __name__ == "__main__":
    print("=== Exporting Champion MultiTask UNet++ to ONNX ===")
    export_and_verify(
        checkpoint_path="models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth",
        onnx_path="models/saved/floorplan_unetpp/floorplan_multitask.onnx"
    )

    print("\n=== Exporting Fallback Router (Blend alpha=0.3) to ONNX ===")
    export_and_verify(
        checkpoint_path="models/saved/floorplan_unetpp/multitask_resnet34_blend_alpha_0_3.pth",
        onnx_path="models/saved/floorplan_unetpp/floorplan_fallback.onnx"
    )
