
from pathlib import Path
from agentic_vision import analyze_video_agentically
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "gymnalyze_mobilenetv3.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

FRAME_INTERVAL = 15
PERSON_CONFIDENCE = 0.25


# --------------------------------------------------
# LOAD YOUR TRAINED MOBILENETV3 MODEL
# --------------------------------------------------

def load_classifier():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Copy gymnalyze_mobilenetv3.pth into the models folder."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    if checkpoint.get("architecture") != "mobilenet_v3_small":
        raise ValueError(
            "The checkpoint architecture is not MobileNetV3-Small."
        )

    classifier = models.mobilenet_v3_small(weights=None)

    in_features = classifier.classifier[-1].in_features
    classifier.classifier[-1] = nn.Linear(
        in_features, len(class_names)
    )

    classifier.load_state_dict(
        checkpoint["model_state_dict"]
    )
    classifier.to(DEVICE)
    classifier.eval()

    preprocessing = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return classifier, class_names, preprocessing


model, class_names, eval_transform = load_classifier()

# YOLO is separate from the classifier. Its purpose here is
# to detect people and draw bounding boxes.
person_detector = YOLO("yolo11n.pt")


# --------------------------------------------------
# CLASSIFY ONE FRAME
# --------------------------------------------------

def predict_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(
        frame_bgr, cv2.COLOR_BGR2RGB
    )
    image = Image.fromarray(frame_rgb)

    tensor = eval_transform(image).unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    return probabilities.cpu().numpy()


# --------------------------------------------------
# PROCESS AN UPLOADED VIDEO
# --------------------------------------------------

def process_video(
    input_path,
    output_path,
    progress_callback=None,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    # PASS 1: Sample frames and predict the overall class.    
    agent_result = analyze_video_agentically(
        video_path=input_path,
        predict_frame=predict_frame,
        class_names=class_names,
        initial_samples=8,
        extra_samples=12,
        confidence_threshold=0.70,
        agreement_threshold=0.65,
    )

    predicted_class = agent_result["predicted_class"]
    confidence = agent_result["confidence"]

    mean_probabilities = np.array([
        agent_result["class_scores"][str(name)]
        for name in class_names
    ])

    if progress_callback:
        progress_callback(0.4)

    # PASS 2: Detect people, draw boxes and write the video.
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(
            "Could not reopen the video for annotation."
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not create the output MP4. "
            "Check your OpenCV video codec installation."
        )

    frames_written = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = person_detector.predict(
                source=frame,
                classes=[0],
                conf=PERSON_CONFIDENCE,
                verbose=False,
            )[0]

            if detections.boxes is not None:
                for box in detections.boxes:
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].cpu().numpy(),
                    )

                    person_score = float(
                        box.conf[0].cpu().item()
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    label = (
                        f"Person {person_score:.2f} | "
                        f"{predicted_class}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # Overall classification banner.
            banner = (
                f"{predicted_class.upper()} "
                f"({confidence:.1%} model score)"
            )

            cv2.rectangle(
                frame,
                (0, 0),
                (min(width, 650), 45),
                (20, 20, 20),
                -1,
            )

            cv2.putText(
                frame,
                banner,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frames_written += 1

            if progress_callback and total_frames > 0:
                fraction = frames_written / total_frames
                progress_callback(0.4 + 0.6 * fraction)

    finally:
        cap.release()
        writer.release()

    if (
        not output_path.exists()
        or output_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "The annotated output video was not created."
        )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_scores": {
            str(name): float(score)
            for name, score in zip(
                class_names, mean_probabilities
            )
        },
        "frames_processed": agent_result["frames_processed"],
        "frames_written": frames_written,
        "output_path": str(output_path),
        "retry_used": agent_result["retry_used"],
        "frame_agreement": agent_result["frame_agreement"],
        "needs_review": agent_result["needs_review"],
        "decision_reason": agent_result["reason"],
    }
