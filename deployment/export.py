from pathlib import Path
import sys
import torch
import onnx
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
M3_ROOT = REPO_ROOT / "M3-P4"

sys.path.insert(0, str(M3_ROOT / "models" / "cnn"))

from model import SpeechEnhancementCNN


MODEL_PATH = M3_ROOT / "model" / "trained_model.pth"
OUTPUT_DIR = REPO_ROOT / "deployment"
ONNX_PATH = OUTPUT_DIR / "m3_speech_enhancement.onnx"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("M3-P6 ACTUAL ONNX EXPORT")
    print("=" * 70)

    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Output:     {ONNX_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model: {MODEL_PATH}"
        )

    # Load the exact architecture used by M3-P4.
    model = SpeechEnhancementCNN()
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Parameters: {parameter_count}")

    # The trained M3 CNN operates on:
    # [batch, channel, frequency, time]
    #
    # Frequency bins = 512 / 2 + 1 = 257.
    dummy_input = torch.randn(
        1,
        1,
        257,
        100,
        dtype=torch.float32,
    )

    with torch.no_grad():
        pytorch_output = model(dummy_input)

    print(
        "PyTorch input shape :",
        tuple(dummy_input.shape),
    )
    print(
        "PyTorch output shape:",
        tuple(pytorch_output.shape),
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["input"],
        output_names=["mask"],
        dynamic_axes={
            "input": {
                0: "batch",
                3: "time",
            },
            "mask": {
                0: "batch",
                3: "time",
            },
        },
        opset_version=18,
        do_constant_folding=True,
    )

    print()
    print("ONNX export completed.")

    # ------------------------------------------------------------
    # ONNX structural validation
    # ------------------------------------------------------------

    onnx_model = onnx.load(str(ONNX_PATH))

    onnx.checker.check_model(onnx_model)

    print("ONNX checker: PASS")

    # ------------------------------------------------------------
    # ONNX Runtime validation
    # ------------------------------------------------------------

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    ort_output = session.run(
        [output_name],
        {
            input_name:
            dummy_input.numpy()
        },
    )[0]

    print(
        "ONNX input shape :",
        session.get_inputs()[0].shape,
    )
    print(
        "ONNX output shape:",
        session.get_outputs()[0].shape,
    )

    # Numerical comparison against PyTorch.
    difference = (
        abs(
            pytorch_output.numpy()
            - ort_output
        )
    )

    max_difference = float(
        difference.max()
    )

    mean_difference = float(
        difference.mean()
    )

    print(
        f"Max PyTorch/ONNX difference : "
        f"{max_difference:.10e}"
    )

    print(
        f"Mean PyTorch/ONNX difference: "
        f"{mean_difference:.10e}"
    )

    if not torch.isfinite(
        torch.from_numpy(ort_output)
    ).all():
        raise RuntimeError(
            "ONNX Runtime produced non-finite output."
        )

    # ------------------------------------------------------------
    # Final artifact information
    # ------------------------------------------------------------

    file_size = ONNX_PATH.stat().st_size

    print()
    print("=" * 70)
    print("M3-P6 EXPORT RESULT")
    print("=" * 70)
    print("Status:              ACCEPTED")
    print(f"Model parameters:    {parameter_count}")
    print(f"ONNX file:           {ONNX_PATH}")
    print(f"ONNX size:           {file_size} bytes")
    print(f"Max numerical diff:  {max_difference:.10e}")
    print(f"Mean numerical diff: {mean_difference:.10e}")
    print()
    print("M3-P6 ACTUAL MODEL EXPORT: PASS")


if __name__ == "__main__":
    main()
