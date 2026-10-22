
import os
import tempfile
import time
import subprocess
from pathlib import Path

import streamlit as st

from gymnalyze_inference import process_video


# FFmpeg executable
FFMPEG_PATH = (
    r"C:\Users\RAJESH M CHAUHAN\Downloads"
    r"\ffmpeg-9.0.2-essentials_build"
    r"\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe"
)

st.set_page_config(
    page_title="Gymnalyze",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Keep video players compact */
    video {
        max-height: 350px !important;
        width: 100% !important;
        object-fit: contain !important;
        [data-testid="stSidebar"] {
         display: none;
        }

        [data-testid="collapsedControl"] {
        display: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Gymnalyze 🎯")
st.caption("AI-powered gymnastics video analysis")

st.write(
    "Upload a gymnastics video to classify the discipline, "
    "detect athletes, and generate an annotated video."
)

uploaded = st.file_uploader(
    "Choose a gymnastics video",
    type=["mp4", "mov", "avi"],
)

if uploaded:
    st.subheader("Input video")

    left, center, right = st.columns([1, 1, 1])

    with center:
        st.video(uploaded, width=400)

    if st.button("Analyze video", type="primary"):

        temp_input = None
        temp_output = None
        temp_browser_output = None

        try:
            # 1. Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded.name).suffix,
                delete=False,
            ) as source:
                source.write(uploaded.getbuffer())
                temp_input = source.name

            # 2. Create temporary paths for both videos
            with tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False,
            ) as output:
                temp_output = output.name

            with tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False,
            ) as output:
                temp_browser_output = output.name

            # 3. Run the existing Gymnalyze inference pipeline
            progress = st.progress(0)
            status = st.empty()
            status.info("Classifying gymnastics video...")

            started = time.perf_counter()

            result = process_video(
                input_path=temp_input,
                output_path=temp_output,
                progress_callback=lambda fraction: progress.progress(
                    max(0.0, min(1.0, float(fraction)))
                ),
            )

            inference_time = time.perf_counter() - started

            if (
                not os.path.exists(temp_output)
                or os.path.getsize(temp_output) == 0
            ):
                raise RuntimeError(
                    "The inference pipeline did not create a valid output video."
                )

            # 4. Convert output to browser-compatible H.264 MP4
            status.info("Converting video for browser playback...")

            conversion = subprocess.run(
                [
                    FFMPEG_PATH,
                    "-y",
                    "-i", temp_output,
                    "-map", "0:v:0",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-an",
                    temp_browser_output,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if conversion.returncode != 0:
                raise RuntimeError(
                    "FFmpeg conversion failed:\n"
                    + conversion.stderr[-3000:]
                )

            if (
                not os.path.exists(temp_browser_output)
                or os.path.getsize(temp_browser_output) == 0
            ):
                raise RuntimeError(
                    "FFmpeg did not create the converted video."
                )

            # 5. Read the converted video into memory
            with open(temp_browser_output, "rb") as video_file:
                annotated_video_bytes = video_file.read()

            # 6. Display results
            progress.progress(1.0)
            status.success("Analysis and video conversion completed!")

            st.divider()
            st.subheader("Analysis results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Predicted discipline",
                    str(result["predicted_class"]).title(),
                )

            with col2:
                st.metric(
                    "Model confidence",
                    f'{float(result["confidence"]):.1%}',
                )

            with col3:
                st.metric(
                    "Inference time",
                    f"{inference_time:.1f} sec",
                )

            st.caption(
                f'Frames processed: {result.get("frames_processed", 0)}'
            )

            st.divider()
            st.subheader("Annotated video")

            left, center, right = st.columns([1, 1, 1])

            with center:
                st.video(
                    annotated_video_bytes,
                    format="video/mp4",
                    width=400,
                )

                st.download_button(
                    label="Download annotated MP4",
                    data=annotated_video_bytes,
                    file_name="gymnalyze_annotated.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

            st.caption(
                "Confidence is the model's prediction score, "
                "not a guarantee that the classification is correct."
            )

        except subprocess.TimeoutExpired:
            st.error(
                "Video conversion timed out. Try a shorter video."
            )

        except FileNotFoundError:
            st.error(
                "FFmpeg was not found. Check that FFMPEG_PATH "
                "points to the actual ffmpeg.exe file."
            )

        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")

        finally:
            # 7. Clean up temporary files
            for path in (
                temp_input,
                temp_output,
                temp_browser_output,
            ):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
