
from pathlib import Path

import cv2

from gymnalyze_inference import (
    predict_frame,
    class_names,
)
from agentic_vision import analyze_video_agentically


video_path = Path("test.mp4")

if not video_path.exists():
    raise FileNotFoundError(
        f"Place test.mp4 in the project root: {video_path.resolve()}"
    )

result = analyze_video_agentically(
    video_path=video_path,
    predict_frame=predict_frame,
    class_names=class_names,
    initial_samples=8,
    extra_samples=12,
    confidence_threshold=0.70,
    agreement_threshold=0.65,
)

print("\n--- GYMNALYZE AGENTIC TEST ---")
for key, value in result.items():
    print(f"{key}: {value}")
