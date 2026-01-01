
import argparse
from pathlib import Path
import sys
import torch
import onnx
import onnxruntime as ort
import numpy as np

# Add src to path so we can import model builder
sys.path.append(str(Path(__file__).parent.parent / "src"))
from model import build_model

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best.ckpt")
    ap.add_argument("--out", default=None, help="Output onnx path (default: same dir as ckpt)")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--check", action="store_true", help="Run onnxruntime check")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Extract metadata
    backbone = checkpoint["backbone"]
    label2id = checkpoint["label2id"]
    img_size = checkpoint["img_size"]
    num_classes = len(label2id)
    state_dict = checkpoint["state_dict"]

    print(f"[INFO] Backbone: {backbone}, Num Classes: {num_classes}, Img Size: {img_size}")

    # Build model
    model = build_model(backbone, num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)

    # Output path
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = ckpt_path.with_name("model.onnx")

    print(f"[INFO] Exporting to: {out_path}")
    
    # Dynamic axes for batch size
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("[SUCCESS] Export complete.")

    if args.check:
        print("[INFO] Validating with ONNX Runtime...")
        # Verify structure
        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)

        # Verify output numerical match
        ort_session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

        # compare torch vs onnx
        torch_out = model(dummy_input)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
        ort_outs = ort_session.run(None, ort_inputs)

        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("[SUCCESS] ONNX Runtime output matches PyTorch output!")

if __name__ == "__main__":
    main()
