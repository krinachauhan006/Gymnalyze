"""
live_camera.py — Isolated Live Camera Inference Module for Gymnalyze.

This module reuses the existing MobileNetV3 classifier and YOLO11n person detector
from gymnalyze_inference.py without modifying any protected inference or model files.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Dict, Any, Tuple, Optional, List, Union

import cv2
import numpy as np

# Import protected inference components safely without modification
from gymnalyze_inference import (
    predict_frame,
    class_names,
    person_detector,
    PERSON_CONFIDENCE,
    DEVICE,
)

BASE_DIR = Path(__file__).resolve().parent
TEST_VIDEOS_DIR = BASE_DIR / "test_videos"

# Detect FFmpeg executable
FFMPEG_PATH = (
    shutil.which("ffmpeg")
    or r"C:\Users\RAJESH M CHAUHAN\Downloads"
       r"\ffmpeg-9.0.2-essentials_build"
       r"\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe"
)


def get_ffmpeg_path() -> str:
    """Return the absolute path to a functional FFmpeg executable."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    fallback = (
        r"C:\Users\RAJESH M CHAUHAN\Downloads"
        r"\ffmpeg-9.0.2-essentials_build"
        r"\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe"
    )
    if Path(fallback).is_file():
        return fallback
    return "ffmpeg"


def get_sample_videos() -> Dict[str, str]:
    """Return available sample video paths for testing or simulation."""
    samples = {}
    if TEST_VIDEOS_DIR.exists():
        for f in TEST_VIDEOS_DIR.glob("*.mp4"):
            label = f.stem
            if "1A" in label:
                display = "Sample 1: Artistic Gymnastics (Floor/Vault)"
            elif "2R" in label:
                display = "Sample 2: Rhythmic Gymnastics (Hoop/Ribbon)"
            elif "3A" in label:
                display = "Sample 3: Artistic Gymnastics (Bars)"
            elif "4A" in label:
                display = "Sample 4: Artistic Gymnastics (Beam)"
            else:
                display = f"Sample: {f.name}"
            samples[display] = str(f)
    root_test = BASE_DIR / "test.mp4"
    if root_test.exists() and "Sample 2: Rhythmic Gymnastics (Hoop/Ribbon)" not in samples:
        samples["Default Test Video (Rhythmic)"] = str(root_test)
    return samples


def check_video_athlete_presence(video_path: str, num_samples: int = 10) -> Tuple[float, int]:
    """
    Check the ratio of sampled frames in a video that contain detected human athletes.
    Used to verify human gymnastics footage and reject cartoon/non-human videos.
    Returns: (athlete_ratio, total_samples)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0, 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0.0, 0

    indices = np.linspace(0, total_frames - 1, min(total_frames, num_samples), dtype=int)
    athlete_frames = 0
    samples_checked = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        samples_checked += 1
        dets = person_detector.predict(source=frame, classes=[0], conf=PERSON_CONFIDENCE, verbose=False)[0]
        if dets.boxes is not None and len(dets.boxes) > 0:
            athlete_frames += 1

    cap.release()
    ratio = athlete_frames / max(1, samples_checked)
    return ratio, samples_checked


def check_camera_available(camera_index: int = 0) -> Tuple[bool, str]:
    """
    Test whether a local camera device is accessible.
    
    Returns:
        (is_available, message)
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return False, (
                f"Camera device {camera_index} is unavailable or access was denied. "
                "Ensure your webcam is plugged in, not currently in use by another application "
                "(such as Teams, Zoom, or another browser window), and camera permissions are granted in Windows Settings."
            )

        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None or frame.size == 0:
            return False, (
                f"Camera device {camera_index} connected but returned an empty frame. "
                "The webcam might still be initializing or blocked by an active privacy shutter."
            )

        h, w = frame.shape[:2]
        return True, f"Webcam ready ({w}x{h} px at index {camera_index})"
    except Exception as exc:
        return False, f"Failed to access camera {camera_index}: {str(exc)}"


def annotate_frame_with_predictions(
    frame_bgr: np.ndarray,
    predicted_class: str,
    confidence: float,
    class_scores: Optional[Dict[str, float]] = None,
    draw_person_boxes: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Annotate a single frame with YOLO person detection bounding boxes and
    the MobileNetV3 discipline classification banner, matching the visual
    logic of gymnalyze_inference.py.
    
    Returns:
        (annotated_frame_bgr, person_count)
    """
    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]
    person_count = 0

    if draw_person_boxes and person_detector is not None:
        try:
            detections = person_detector.predict(
                source=annotated,
                classes=[0],
                conf=PERSON_CONFIDENCE,
                verbose=False,
            )[0]

            if detections.boxes is not None:
                person_count = len(detections.boxes)
                for box in detections.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    person_score = float(box.conf[0].cpu().item())

                    # Draw neon-green bounding box for athlete
                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    if predicted_class and predicted_class.lower() != "none" and confidence > 0.0:
                        label = f"Athlete {person_score:.2f} | {predicted_class.title()}"
                    else:
                        label = f"Athlete {person_score:.2f}"

                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
        except Exception:
            pass

    # Overall classification banner (matching gymnalyze_inference.py)
    if predicted_class and predicted_class.lower() != "none" and confidence > 0.0:
        banner = f"{predicted_class.upper()} ({confidence:.1%} model score)"
    else:
        banner = "ANALYZING STREAM (None, 0.0%)"
    banner_w = min(width, 540)

    # Semi-transparent dark background for banner
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (banner_w, 42), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    # Accent border on banner
    cv2.line(annotated, (0, 42), (banner_w, 42), (74, 222, 128), 2)

    # Banner text
    cv2.putText(
        annotated,
        banner,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (74, 222, 128),
        2,
        cv2.LINE_AA,
    )

    return annotated, person_count


def transcode_to_h264(raw_mp4_path: str, web_mp4_path: str) -> None:
    """Transcode a raw MP4 video into web-compatible H.264 format using FFmpeg."""
    ffmpeg = get_ffmpeg_path()
    if not Path(ffmpeg).is_file() and shutil.which(ffmpeg) is None:
        raise FileNotFoundError(f"FFmpeg executable not found at: {ffmpeg}")

    conversion = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i", raw_mp4_path,
            "-map", "0:v:0",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            web_mp4_path,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if conversion.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {conversion.stderr[-1000:]}")

    if not Path(web_mp4_path).exists() or Path(web_mp4_path).stat().st_size == 0:
        raise RuntimeError("FFmpeg did not produce a valid converted video.")


class LiveCameraSession:
    """
    Manages a live camera or simulated stream analysis session, tracking rolling probabilities,
    detecting athletes with YOLO, annotating frames in real time, and recording
    an annotated MP4 video.
    """

    def __init__(self, source: Union[int, str] = 0, sample_interval: int = 2):
        self.source = source
        self.sample_interval = sample_interval  # Run MobileNetV3 every N frames for responsive FPS
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.temp_raw_file: Optional[str] = None
        self.temp_web_file: Optional[str] = None

        self.frames_captured = 0
        self.frames_analyzed = 0
        self.athlete_frames = 0
        self.consecutive_empty_frames = 0
        self.all_probabilities: List[np.ndarray] = []
        self.latest_probabilities = np.array([0.0, 0.0])
        self.latest_class = "None"
        self.latest_confidence = 0.0
        self.latest_persons = 0
        self.start_time = 0.0

    def start(self, width: int = 640, height: int = 480, fps: float = 24.0) -> None:
        """Initialize camera/stream and temporary video writer."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as raw_f:
            self.temp_raw_file = raw_f.name

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as web_f:
            self.temp_web_file = web_f.name

        self.writer = cv2.VideoWriter(
            self.temp_raw_file,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (actual_w, actual_h),
        )

        self.start_time = time.perf_counter()
        self.frames_captured = 0
        self.frames_analyzed = 0
        self.athlete_frames = 0
        self.consecutive_empty_frames = 0
        self.all_probabilities = []

    def read_and_process_frame(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Read the next frame from webcam or simulated source, run YOLO + MobileNetV3 inference,
        annotate it, write to video, and return frame + live statistics.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None, {"error": "Camera is closed or disconnected"}

        ok, frame = self.cap.read()
        if not ok or frame is None or frame.size == 0:
            self.consecutive_empty_frames += 1
            # If simulated video reached the end, loop back
            if isinstance(self.source, str) and self.consecutive_empty_frames <= 3:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok or frame is None:
                return False, None, {"error": "Camera stream ended or dropped frames"}

        self.consecutive_empty_frames = 0
        self.frames_captured += 1

        # Check for person/athlete presence in frame first
        person_count = 0
        if person_detector is not None:
            try:
                dets = person_detector.predict(
                    source=frame,
                    classes=[0],
                    conf=PERSON_CONFIDENCE,
                    verbose=False,
                )[0]
                if dets.boxes is not None:
                    person_count = len(dets.boxes)
            except Exception:
                pass

        if person_count > 0:
            self.athlete_frames += 1
            # Run MobileNetV3 classification on sample interval
            if self.frames_captured == 1 or (self.frames_captured % self.sample_interval == 0):
                try:
                    probs = predict_frame(frame)
                    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                        best_idx = int(np.argmax(probs))
                        best_conf = float(probs[best_idx])

                        # Only classify as a gymnastics discipline if certainty threshold is met (>= 0.65)
                        # Prevents false positive classification when simply sitting or not performing gymnastics
                        if best_conf >= 0.65:
                            self.latest_probabilities = probs
                            self.all_probabilities.append(probs)
                            self.frames_analyzed += 1
                            self.latest_class = str(class_names[best_idx])
                            self.latest_confidence = best_conf
                        else:
                            # Person is in frame but posture is neutral / not performing gymnastics
                            self.latest_class = "None"
                            self.latest_confidence = 0.0
                            self.latest_probabilities = np.array([0.0, 0.0])
                except Exception:
                    pass
        else:
            # No athlete / person in frame (e.g. empty camera, background only)
            # Default to None and 0%
            self.latest_class = "None"
            self.latest_confidence = 0.0
            self.latest_probabilities = np.array([0.0, 0.0])

        # Annotate with YOLO person boxes and classification banner
        annotated, person_count = annotate_frame_with_predictions(
            frame_bgr=frame,
            predicted_class=self.latest_class,
            confidence=self.latest_confidence,
            class_scores={
                str(name): float(score)
                for name, score in zip(class_names, self.latest_probabilities)
            },
            draw_person_boxes=True,
        )

        self.latest_persons = person_count

        # Write annotated frame to video writer
        if self.writer is not None and self.writer.isOpened():
            self.writer.write(annotated)

        # Calculate live rolling statistics
        elapsed = max(0.001, time.perf_counter() - self.start_time)
        fps = self.frames_captured / elapsed

        if self.all_probabilities and self.athlete_frames > 0:
            mean_probs = np.mean(self.all_probabilities, axis=0)
            overall_idx = int(np.argmax(mean_probs))
            overall_conf = float(mean_probs[overall_idx])
            if overall_conf >= 0.65:
                overall_class = str(class_names[overall_idx])
            else:
                overall_class = "None"
                overall_conf = 0.0
                mean_probs = np.array([0.0, 0.0])
        else:
            overall_class = self.latest_class
            overall_conf = self.latest_confidence
            mean_probs = self.latest_probabilities

        stats = {
            "predicted_class": overall_class,
            "confidence": overall_conf,
            "latest_class": self.latest_class,
            "latest_confidence": self.latest_confidence,
            "class_scores": {
                str(name): float(score)
                for name, score in zip(class_names, mean_probs)
            },
            "latest_scores": {
                str(name): float(score)
                for name, score in zip(class_names, self.latest_probabilities)
            },
            "frames_captured": self.frames_captured,
            "frames_analyzed": self.frames_analyzed,
            "athletes_detected": person_count,
            "athlete_frames": self.athlete_frames,
            "fps": fps,
            "elapsed": elapsed,
        }

        # Convert to RGB for Streamlit rendering
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return True, frame_rgb, stats

    def stop_and_finalize(self) -> Dict[str, Any]:
        """
        Close camera and writer, transcode recorded video to web MP4,
        and compute final session metrics.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        elapsed = max(0.001, time.perf_counter() - self.start_time)

        if self.all_probabilities and self.athlete_frames > 0:
            mean_probs = np.mean(self.all_probabilities, axis=0)
            best_idx = int(np.argmax(mean_probs))
            final_conf = float(mean_probs[best_idx])

            if final_conf >= 0.65:
                final_class = str(class_names[best_idx])
                # Frame agreement
                labels = [int(np.argmax(p)) for p in self.all_probabilities]
                agreement = sum(idx == best_idx for idx in labels) / len(labels)
                needs_review = (final_conf < 0.70) or (agreement < 0.65)
                reason = (
                    "Live camera frame-by-frame analysis completed. "
                    + ("Confidence and agreement met configured thresholds." if not needs_review else "Prediction remains uncertain; human review advised.")
                )
            else:
                final_class = "None (No Routine Detected)"
                final_conf = 0.0
                agreement = 0.0
                needs_review = True
                mean_probs = np.array([0.0, 0.0])
                reason = "No confident gymnastics routine movements detected during this camera session."
        else:
            mean_probs = np.array([0.0, 0.0])
            final_class = "None (No Gymnastics Detected)"
            final_conf = 0.0
            agreement = 0.0
            needs_review = True
            reason = "No athlete or gymnastics routine was detected in the camera stream."

        web_video_bytes = None
        if self.temp_raw_file and Path(self.temp_raw_file).exists() and Path(self.temp_raw_file).stat().st_size > 0:
            try:
                transcode_to_h264(self.temp_raw_file, self.temp_web_file)
                if Path(self.temp_web_file).exists():
                    with open(self.temp_web_file, "rb") as f:
                        web_video_bytes = f.read()
            except Exception as exc:
                print(f"[live_camera] Transcode error: {exc}")

        return {
            "predicted_class": final_class,
            "confidence": final_conf,
            "class_scores": {
                str(name): float(score)
                for name, score in zip(class_names, mean_probs)
            },
            "frames_processed": self.frames_analyzed,
            "frames_captured": self.frames_captured,
            "athlete_frames": self.athlete_frames,
            "frame_agreement": agreement,
            "needs_review": needs_review,
            "duration": elapsed,
            "video_bytes": web_video_bytes,
            "decision_reason": (
                "Live camera frame-by-frame analysis completed. "
                + ("Confidence and agreement met configured thresholds." if not needs_review else "Prediction remains uncertain; human review advised.")
            ),
        }

    def cleanup(self) -> None:
        """Remove any temporary files created during session."""
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        for path in (self.temp_raw_file, self.temp_web_file):
            if path and Path(path).exists():
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass
