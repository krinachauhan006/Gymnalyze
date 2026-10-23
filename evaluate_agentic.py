
import csv
import time
from pathlib import Path

from gymnalyze_inference import predict_frame, class_names
from agentic_vision import analyze_video_agentically


# Add your own labeled test videos here.
# Use the exact class names from your checkpoint.
TEST_VIDEOS = [
    {"path": "test_videos/test1A.mp4", "true_label": "artistic"},
     {"path": "test_videos/test2R.mp4", "true_label": "rhythmic"},
     {"path": "test_videos/test3A.mp4", "true_label": "artistic"},
]

output_csv = Path("agentic_evaluation.csv")
rows = []

for item in TEST_VIDEOS:
    video_path = Path(item["path"])

    if not video_path.exists():
        print(f"Skipping missing video: {video_path}")
        continue

    start = time.perf_counter()

    result = analyze_video_agentically(
        video_path=video_path,
        predict_frame=predict_frame,
        class_names=class_names,
        initial_samples=8,
        extra_samples=12,
        confidence_threshold=0.70,
        agreement_threshold=0.65,
    )

    elapsed = time.perf_counter() - start
    true_label = item["true_label"]
    predicted = result["predicted_class"]

    row = {
        "video": str(video_path),
        "true_label": true_label,
        "predicted_label": predicted,
        "correct": predicted == true_label,
        "confidence": result["confidence"],
        "frame_agreement": result["frame_agreement"],
        "frames_processed": result["frames_processed"],
        "retry_used": result["retry_used"],
        "needs_review": result["needs_review"],
        "processing_seconds": round(elapsed, 3),
    }
    rows.append(row)

    print(
        f"{video_path}: true={true_label}, "
        f"predicted={predicted}, "
        f"correct={row['correct']}, "
        f"review={row['needs_review']}, "
        f"time={elapsed:.2f}s"
    )

if rows:
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    accuracy = sum(row["correct"] for row in rows) / len(rows)

    print(f"\nVideos evaluated: {len(rows)}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Results saved to: {output_csv.resolve()}")
else:
    print("No videos were evaluated. Check your video paths.")
