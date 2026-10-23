
from pathlib import Path

from collections import Counter

import cv2
import numpy as np


def _sample_positions(total_frames, count):
    """Return unique, evenly distributed frame indices."""
    if total_frames <= 0:
        return []

    count = min(total_frames, count)
    return np.linspace(
        0, total_frames - 1, num=count, dtype=int
    ).tolist()


def analyze_video_agentically(
    video_path,
    predict_frame,
    class_names,
    initial_samples=8,
    extra_samples=12,
    confidence_threshold=0.70,
    agreement_threshold=0.65,
):
    """
    Reuse the existing model's predict_frame(frame_bgr).

    predict_frame returns a NumPy array containing probabilities
    in exactly the same order as class_names.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise RuntimeError("The video has no readable frames.")

        initial_positions = _sample_positions(
            total_frames, initial_samples
        )

        all_positions = _sample_positions(
            total_frames, initial_samples + extra_samples
        )

        positions_to_process = list(initial_positions)
        predictions = []

        def read_predictions(positions):
            batch = []

            for position in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = cap.read()

                if not ok or frame is None:
                    continue

                probabilities = np.asarray(
                    predict_frame(frame), dtype=np.float64
                ).reshape(-1)

                if len(probabilities) != len(class_names):
                    raise ValueError(
                        "Model output count does not match class_names."
                    )

                if (
                    not np.isfinite(probabilities).all()
                    or np.any(probabilities < 0)
                    or probabilities.sum() <= 0
                ):
                    raise ValueError(
                        "Model returned invalid class probabilities."
                    )

                probabilities = probabilities / probabilities.sum()
                batch.append(probabilities)

            return batch

        predictions = read_predictions(initial_positions)

        if not predictions:
            raise RuntimeError("No frames could be classified.")

        def summarize(batch):
            mean_scores = np.mean(batch, axis=0)
            best_index = int(np.argmax(mean_scores))

            frame_labels = [
                int(np.argmax(scores)) for scores in batch
            ]

            agreement = (
                sum(index == best_index for index in frame_labels)
                / len(frame_labels)
            )

            return best_index, mean_scores, agreement

        best_index, mean_scores, agreement = summarize(predictions)

        retry_used = (
            float(mean_scores[best_index]) < confidence_threshold
            or agreement < agreement_threshold
        )

        if retry_used:
            initial_set = set(initial_positions)
            extra_positions_to_process = [
                position for position in all_positions
                if position not in initial_set
            ]

            extra_predictions = read_predictions(
                extra_positions_to_process
            )
            predictions.extend(extra_predictions)

            best_index, mean_scores, agreement = summarize(
                predictions
            )

        confidence = float(mean_scores[best_index])
        needs_review = (
            confidence < confidence_threshold
            or agreement < agreement_threshold
        )

        if needs_review:
            reason = (
                "The prediction remains uncertain after "
                "additional frame sampling."
            )
        elif retry_used:
            reason = (
                "Additional frames were analyzed; the final "
                "prediction met the configured thresholds."
            )
        else:
            reason = (
                "The initial frame sample met the configured thresholds."
            )

        return {
            "predicted_class": str(class_names[best_index]),
            "confidence": confidence,
            "class_scores": {
                str(name): float(score)
                for name, score in zip(class_names, mean_scores)
            },
            "frames_processed": len(predictions),
            "retry_used": retry_used,
            "frame_agreement": float(agreement),
            "needs_review": needs_review,
            "reason": reason,
        }

    finally:
        cap.release()
