
from pathlib import Path
import torch

checkpoint_path = Path("models/gymnalyze_mobilenetv3.pth")

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path.resolve()}\n"
        "Update checkpoint_path to your actual model location."
    )

# Only load a checkpoint you trust, such as the one you trained.
try:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
except Exception as exc:
    raise RuntimeError(
        "Could not load this checkpoint with weights_only=True. "
        "Inspect the checkpoint format before changing the loading options."
    ) from exc

print("Checkpoint:", checkpoint_path.resolve())
print("Loaded object type:", type(checkpoint).__name__)

if isinstance(checkpoint, dict):
    print("Top-level keys:", list(checkpoint.keys()))

    for key in ("architecture", "class_names", "image_size"):
        if key in checkpoint:
            print(f"{key}: {checkpoint[key]}")

    if "model_state_dict" in checkpoint:
        print("Model weights key: model_state_dict")
    elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        print("The checkpoint appears to be a raw state dictionary.")
    else:
        print("Inspect the printed keys to identify the saved model weights.")
else:
    print("This is not a dictionary checkpoint; inspect its saved format.")
