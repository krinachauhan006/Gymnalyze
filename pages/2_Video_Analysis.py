"""
pages/2_Video_Analysis.py — Interactive AI Gymnastics Analysis Studio for Gymnalyze.
Supports:
  Mode A: Live Camera Analysis (Real-time webcam inference, YOLO athlete detection, MP4 recording)
  Mode B: Upload Video Analysis (Multi-pass agentic sampling, MobileNetV3 classification, annotated MP4)
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import numpy as np
import cv2

# Import protected inference components without modification
from gymnalyze_inference import (
    process_video,
    class_names,
    predict_frame,
    person_detector,
    PERSON_CONFIDENCE,
)

# Import isolated live camera and UI helper modules
import live_camera
from live_camera import (
    check_camera_available,
    LiveCameraSession,
    get_ffmpeg_path,
    get_sample_videos,
    check_video_athlete_presence,
)
import ui_components
from ui_components import (
    get_custom_css,
    render_brand_header,
    render_status_pill,
    render_probability_bars,
    render_agentic_explanation,
    render_review_alert,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FFMPEG_PATH = get_ffmpeg_path()

# --------------------------------------------------
# PAGE CONFIGURATION & STYLES
# --------------------------------------------------

st.set_page_config(
    page_title="Gymnalyze | AI Gymnastics Analysis Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(get_custom_css(), unsafe_allow_html=True)

# --------------------------------------------------
# HEADER & TOP NAVIGATION
# --------------------------------------------------

nav_left, nav_right = st.columns([4, 1])
with nav_left:
    st.markdown(
        render_brand_header(subtitle="Interactive AI Gymnastics Analysis Studio"),
        unsafe_allow_html=True,
    )
with nav_right:
    st.write("")
    if st.button("← Return to Overview", width="stretch"):
        st.switch_page("app.py")

# --------------------------------------------------
# MODE SELECTION (Two prominent modes)
# --------------------------------------------------

# Determine initial mode from session state or default
default_mode_idx = 0
if st.session_state.get("selected_mode") == "upload":
    default_mode_idx = 1
elif st.session_state.get("selected_mode") == "live":
    default_mode_idx = 0

mode_choice = st.radio(
    "Select Analysis Mode",
    options=["🎥 Mode A: Live Camera Analysis", "📁 Mode B: Upload Video Analysis"],
    index=default_mode_idx,
    horizontal=True,
    label_visibility="collapsed",
)

st.write("")

# ==============================================================================
# MODE A: LIVE CAMERA ANALYSIS
# ==============================================================================
if "Mode A" in mode_choice:
    st.markdown(
        """
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 14px; padding: 18px 24px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h2 style="margin: 0; font-size: 1.4rem; font-weight: 800; color: #ffffff;">
                        🎥 Mode A — Live Camera Analysis
                    </h2>
                    <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                        Perform real-time gymnastics discipline analysis using your live webcam or interactive camera simulation.
                    </p>
                </div>
                <div>
                    <span class="status-pill ready">Real-Time YOLO11n + MobileNetV3</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check local webcam availability only when Mode A is active
    cam_index = 0
    camera_is_ready, camera_message = check_camera_available(cam_index)

    # Mode A Settings & Camera Source Selection
    col_ctrl, col_info = st.columns([1.2, 1], gap="medium")

    with col_ctrl:
        st.subheader("Camera Setup & Controls")

        source_type = st.radio(
            "Video Input Source",
            options=["Hardware Webcam (Live Feed)", "Simulated Camera (Sample Gymnastics Routine)"],
            index=0 if camera_is_ready else 1,
            horizontal=True,
        )

        sample_videos = get_sample_videos()
        selected_sim_path = None
        if "Simulated" in source_type:
            selected_sample_label = st.selectbox(
                "Choose Simulated Routine",
                options=list(sample_videos.keys()),
                index=0,
            )
            selected_sim_path = sample_videos[selected_sample_label]
            st.caption(f"Using test video source: {Path(selected_sim_path).name}")

        # Duration & performance settings
        cfg1, cfg2 = st.columns(2)
        with cfg1:
            session_duration = st.slider(
                "Analysis Duration (Seconds)",
                min_value=3,
                max_value=30,
                value=8,
                step=1,
                help="Duration of the live analysis recording session.",
            )
        with cfg2:
            sample_step = st.selectbox(
                "Classifier Sampling Step",
                options=[1, 2, 3],
                index=1,
                format_func=lambda x: f"Every {x} frame{'s' if x>1 else ''} (Balanced FPS)" if x==2 else (f"Every frame (High Precision)" if x==1 else f"Every 3 frames (High Speed)"),
                help="How frequently to evaluate the MobileNetV3 classifier. YOLO person tracking runs on all frames.",
            )

    with col_info:
        st.subheader("Stream Diagnostic Status")
        if "Hardware" in source_type:
            if camera_is_ready:
                st.markdown(
                    render_status_pill("Camera Ready — Hardware Accessible", "ready"),
                    unsafe_allow_html=True,
                )
                st.info(f"✅ {camera_message}")
            else:
                st.markdown(
                    render_status_pill("Camera Unavailable / Access Denied", "active"),
                    unsafe_allow_html=True,
                )
                st.error(
                    f"⚠️ {camera_message}\n\n"
                    "**Tip:** You can switch to 'Simulated Camera' above to run the live analysis pipeline immediately without hardware access."
                )
        else:
            st.markdown(
                render_status_pill("Simulation Source Active", "processing"),
                unsafe_allow_html=True,
            )
            st.success("✅ Demo routine loaded. Ready to stream into real-time inference pipeline.")

        st.caption(
            "Live mode runs YOLO11n person detection on each frame and periodically evaluates "
            "MobileNetV3-Small to maintain real-time tracking while recording an annotated video."
        )

    st.divider()

    # Action Buttons: Start Live Analysis / Reset
    btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1, 2])
    with btn_col1:
        start_analysis = st.button("▶ Start Live Analysis", type="primary", width="stretch")
    with btn_col2:
        reset_analysis = st.button("🔄 Reset Live Session", width="stretch")

    if reset_analysis:
        st.session_state.pop("live_session_results", None)
        st.rerun()

    # Active Live Analysis Viewport
    feed_col, metrics_col = st.columns([1.3, 1], gap="medium")

    feed_placeholder = feed_col.empty()
    status_placeholder = feed_col.empty()
    progress_bar = feed_col.progress(0.0)

    metric_class_ph = metrics_col.empty()
    metric_conf_ph = metrics_col.empty()
    prob_bars_ph = metrics_col.empty()
    metric_stats_ph = metrics_col.empty()

    # Initial idle state placeholder
    if not start_analysis and "live_session_results" not in st.session_state:
        feed_placeholder.markdown(
            """
            <div style="background: #0b1120; border: 2px dashed #1e293b; border-radius: 16px; height: 380px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 20px;">
                <div style="font-size: 3.5rem; margin-bottom: 12px;">📹</div>
                <h3 style="color: #ffffff; margin: 0 0 8px 0;">Live Camera Feed Viewport</h3>
                <p style="color: #94a3b8; max-width: 440px; margin: 0; font-size: 0.9rem;">
                    Click <strong>'Start Live Analysis'</strong> above to begin streaming frames, detecting athletes, and analyzing gymnastics moves in real time.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_class_ph.metric(
            label="Live Predicted Discipline",
            value="None",
            delta="Awaiting live stream...",
        )
        metric_conf_ph.metric(
            label="Model Prediction Score",
            value="0%",
            delta="Idle",
        )
        prob_bars_ph.markdown(
            render_probability_bars({"artistic": 0.0, "rhythmic": 0.0}, "None"),
            unsafe_allow_html=True,
        )
        metric_stats_ph.markdown(
            f"""
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 14px; font-size: 0.85rem; color: #94a3b8;">
                <div>⏱ <strong>Elapsed:</strong> 0.0s / {session_duration}s</div>
                <div>🎞 <strong>Frames Captured:</strong> 0 (0 classified)</div>
                <div>🤸 <strong>Athlete Tracking:</strong> Waiting for start...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Run Live Analysis Loop
    if start_analysis:
        video_src: Any = 0 if "Hardware" in source_type else selected_sim_path
        if "Hardware" in source_type and not camera_is_ready:
            st.error("Cannot start live analysis: hardware camera is unavailable. Please select 'Simulated Camera' to proceed.")
        else:
            session = LiveCameraSession(source=video_src, sample_interval=sample_step)
            try:
                status_placeholder.markdown(
                    render_status_pill("Recording & Analyzing Live Routine...", "active"),
                    unsafe_allow_html=True,
                )
                session.start()

                target_frames = int(session_duration * 20)  # ~20 FPS target
                start_t = time.perf_counter()

                for frame_idx in range(1, target_frames + 1):
                    ok, frame_rgb, stats = session.read_and_process_frame()
                    if not ok:
                        break

                    # Update live feed image
                    feed_placeholder.image(frame_rgb, width="stretch")

                    # Update progress bar
                    current_fraction = min(1.0, frame_idx / target_frames)
                    progress_bar.progress(current_fraction)

                    # Update live metric cards
                    raw_pred = stats.get("predicted_class", "None")
                    pred_class = "None" if (not raw_pred or str(raw_pred).lower() == "none") else str(raw_pred).title()
                    conf_val = float(stats.get("confidence", 0.0))
                    conf_display = f"{conf_val:.1%}" if conf_val > 0.0 else "0%"

                    metric_class_ph.metric(
                        label="Live Predicted Discipline",
                        value=pred_class,
                        delta="Athlete in frame" if stats.get("athletes_detected", 0) > 0 else "Searching for athlete...",
                    )
                    metric_conf_ph.metric(
                        label="Model Prediction Score",
                        value=conf_display,
                        delta=f"{stats.get('fps', 0):.1f} FPS",
                    )

                    # Update dual probability breakdown
                    prob_bars_ph.markdown(
                        render_probability_bars(stats.get("class_scores", {}), stats.get("predicted_class", "")),
                        unsafe_allow_html=True,
                    )

                    metric_stats_ph.markdown(
                        f"""
                        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 14px; font-size: 0.85rem; color: #94a3b8;">
                            <div>⏱ <strong>Elapsed:</strong> {stats.get('elapsed', 0):.1f}s / {session_duration}s</div>
                            <div>🎞 <strong>Frames Captured:</strong> {stats.get('frames_captured', 0)} ({stats.get('frames_analyzed', 0)} classified)</div>
                            <div>🤸 <strong>Athlete Tracking:</strong> {stats.get('athletes_detected', 0)} in view ({stats.get('athlete_frames', 0)} total)</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Check max elapsed time
                    if (time.perf_counter() - start_t) >= session_duration:
                        break

                progress_bar.progress(1.0)
                status_placeholder.markdown(
                    render_status_pill("Processing & Transcoding Recorded Video...", "processing"),
                    unsafe_allow_html=True,
                )

                # Finalize video and compute session statistics
                results = session.stop_and_finalize()
                st.session_state["live_session_results"] = results
                status_placeholder.markdown(
                    render_status_pill("Live Analysis Complete!", "complete"),
                    unsafe_allow_html=True,
                )

            except Exception as exc:
                st.error(f"Live analysis encountered an error: {str(exc)}")
            finally:
                session.cleanup()

    # Render Live Results Dashboard (if session has completed)
    if "live_session_results" in st.session_state:
        res = st.session_state["live_session_results"]
        st.write("")
        st.divider()

        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                <span style="font-size: 1.4rem;">📊</span>
                <h3 style="margin: 0; font-size: 1.35rem; font-weight: 800; color: #ffffff;">
                    Live Camera Analysis Results Dashboard
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Primary Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Predicted Discipline", str(res["predicted_class"]).title())
        with m2:
            st.metric("Model Confidence", f"{float(res['confidence']):.1%}")
        with m3:
            st.metric("Frames Analyzed", f"{res['frames_processed']} / {res.get('frames_captured', 0)}")
        with m4:
            st.metric("Frame Consensus", f"{res.get('frame_agreement', 0):.1%}")

        st.caption(
            "Confidence is the model's prediction score across sampled frames, "
            "not an absolute guarantee of correctness."
        )

        # Review Required Warning
        st.markdown(
            render_review_alert(
                res.get("needs_review", False),
                res.get("decision_reason", ""),
                float(res.get("confidence", 0)),
                float(res.get("frame_agreement", 0)),
            ),
            unsafe_allow_html=True,
        )

        # Probability breakdown
        st.markdown(
            render_probability_bars(res.get("class_scores", {}), res.get("predicted_class", "")),
            unsafe_allow_html=True,
        )

        # Annotated Video Preview & Download
        if res.get("video_bytes") is not None:
            st.write("")
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    <span style="font-size: 1.2rem;">🎬</span>
                    <h4 style="margin: 0; color: #ffffff; font-size: 1.15rem;">
                        Recorded & Annotated Video Preview
                    </h4>
                </div>
                """,
                unsafe_allow_html=True,
            )

            vid_col_left, vid_col_center, vid_col_right = st.columns([1, 2, 1])
            with vid_col_center:
                st.video(res["video_bytes"], format="video/mp4")
                st.download_button(
                    label="⬇  Download Recorded Annotated MP4",
                    data=res["video_bytes"],
                    file_name="gymnalyze_live_annotated.mp4",
                    mime="video/mp4",
                    width="stretch",
                )

# ==============================================================================
# MODE B: UPLOAD VIDEO ANALYSIS
# ==============================================================================
elif "Mode B" in mode_choice:
    st.markdown(
        """
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 14px; padding: 18px 24px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h2 style="margin: 0; font-size: 1.4rem; font-weight: 800; color: #ffffff;">
                        📁 Mode B — Upload Video Analysis
                    </h2>
                    <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                        Analyze high-definition gymnastics videos with multi-pass Agentic Vision frame sampling and YOLO athlete tracking.
                    </p>
                </div>
                <div>
                    <span class="status-pill complete">Multi-Pass Agentic Vision</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_col, preset_col = st.columns([1.5, 1], gap="medium")

    with upload_col:
        uploaded = st.file_uploader(
            "Choose or drag-and-drop a gymnastics video",
            type=["mp4", "mov", "avi", "mkv"],
            help="Supported formats: MP4, MOV, AVI, MKV. Maximum recommended size: 100MB.",
        )

    sample_videos = get_sample_videos()
    with preset_col:
        st.markdown(
            """
            <div style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                Quick Hackathon Demo Presets
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_sample_key = st.selectbox(
            "Select a pre-loaded gymnastics clip",
            options=["None (Use Uploaded File)"] + list(sample_videos.keys()),
            index=0,
            help="Select an existing verified gymnastics clip for instant hackathon testing.",
        )

    # Determine active input file
    active_video_path = None
    active_video_bytes = None
    active_filename = ""
    active_filesize_mb = 0.0

    if uploaded is not None:
        active_video_bytes = uploaded.getvalue()
        active_filename = uploaded.name
        active_filesize_mb = len(active_video_bytes) / (1024 * 1024)
    elif selected_sample_key != "None (Use Uploaded File)":
        active_video_path = sample_videos[selected_sample_key]
        active_filename = Path(active_video_path).name
        active_filesize_mb = Path(active_video_path).stat().st_size / (1024 * 1024)
        with open(active_video_path, "rb") as f:
            active_video_bytes = f.read()

    # Display video preview and details if available
    if active_video_bytes is not None:
        st.markdown(
            f"""
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 12px 18px; margin: 12px 0; display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600; color: #ffffff;">📄 Selected: <code>{active_filename}</code></span>
                <span style="font-family: 'JetBrains Mono', monospace; color: #38bdf8; font-weight: 600;">{active_filesize_mb:.2f} MB</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        prev_left, prev_center, prev_right = st.columns([1, 2, 1])
        with prev_center:
            st.video(active_video_bytes)

        st.write("")
        run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
        with run_col2:
            analyze_button = st.button("🚀  Analyse Gymnastics Video", type="primary", width="stretch")

        if analyze_button:
            temp_input = None
            temp_output = None
            temp_browser_output = None

            try:
                # 1. Write uploaded bytes or copy file to temp source
                suffix = Path(active_filename).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as source:
                    source.write(active_video_bytes)
                    temp_input = source.name

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out_f:
                    temp_output = out_f.name

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as web_f:
                    temp_browser_output = web_f.name

                progress = st.progress(0.0)
                status = st.empty()

                status.markdown(
                    render_status_pill("Verifying Athlete & Gymnastics Content...", "processing"),
                    unsafe_allow_html=True,
                )

                started = time.perf_counter()

                # Check if the video contains actual human athletes or is a cartoon/non-gymnastics video
                athlete_ratio, checked_frames = check_video_athlete_presence(temp_input, num_samples=10)

                if athlete_ratio < 0.20:
                    # Non-gymnastics content detected (e.g. cartoon, animation, non-human video)
                    progress.progress(1.0)
                    inference_time = time.perf_counter() - started
                    status.markdown(
                        render_status_pill("Non-Gymnastics Video Detected", "paused"),
                        unsafe_allow_html=True,
                    )
                    result = {
                        "predicted_class": "None (Non-Gymnastics Video)",
                        "confidence": 0.0,
                        "class_scores": {"artistic": 0.0, "rhythmic": 0.0},
                        "frames_processed": checked_frames,
                        "frames_written": 0,
                        "frame_agreement": 0.0,
                        "needs_review": True,
                        "retry_used": False,
                        "decision_reason": (
                            "No human athletes or gymnasts were detected in this video (e.g. cartoon, animation, or non-gymnastics footage). "
                            "Gymnalyze requires human gymnastics footage to classify disciplines."
                        ),
                    }
                    annotated_video_bytes = active_video_bytes
                else:
                    # Verified human athlete content -> run full agentic process_video
                    status.markdown(
                        render_status_pill("Executing Agentic Multi-Pass Frame Sampling...", "active"),
                        unsafe_allow_html=True,
                    )

                    # Call the existing protected inference pipeline directly
                    result = process_video(
                        input_path=temp_input,
                        output_path=temp_output,
                        progress_callback=lambda fraction: progress.progress(
                            max(0.0, min(1.0, float(fraction)))
                        ),
                    )

                    inference_time = time.perf_counter() - started

                    if not os.path.exists(temp_output) or os.path.getsize(temp_output) == 0:
                        raise RuntimeError("Inference pipeline did not produce a valid output video.")

                    # Convert to browser-compatible H.264 MP4 using FFmpeg
                    status.markdown(
                        render_status_pill("Transcoding to Browser-Compatible H.264...", "processing"),
                        unsafe_allow_html=True,
                    )

                    if not Path(FFMPEG_PATH).is_file() and shutil.which(FFMPEG_PATH) is None:
                        raise FileNotFoundError(f"FFmpeg executable not found: {FFMPEG_PATH}")

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
                        raise RuntimeError("FFmpeg conversion failed:\n" + conversion.stderr[-2000:])

                    with open(temp_browser_output, "rb") as video_file:
                        annotated_video_bytes = video_file.read()

                    progress.progress(1.0)
                    status.markdown(
                        render_status_pill("Analysis & Video Transcoding Complete!", "complete"),
                        unsafe_allow_html=True,
                    )

                # --------------------------------------------------
                # RESULTS DASHBOARD
                # --------------------------------------------------

                st.write("")
                st.divider()

                st.markdown(
                    """
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                        <span style="font-size: 1.4rem;">📊</span>
                        <h3 style="margin: 0; font-size: 1.35rem; font-weight: 800; color: #ffffff;">
                            Video Analysis Results Dashboard
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    raw_pred = str(result["predicted_class"])
                    if raw_pred.lower().startswith("none") or raw_pred.lower() == "none":
                        pred_display = raw_pred
                    else:
                        pred_display = raw_pred.title()
                    st.metric("Predicted Discipline", pred_display)
                with r2:
                    conf_val = float(result.get("confidence", 0.0))
                    conf_display = f"{conf_val:.1%}" if conf_val > 0.0 else "0%"
                    st.metric("Model Confidence", conf_display)
                with r3:
                    st.metric("Frames Analyzed", f"{result['frames_processed']}")
                with r4:
                    agr_val = float(result.get("frame_agreement", 0.0))
                    agr_display = f"{agr_val:.1%}" if agr_val > 0.0 else "0%"
                    st.metric("Frame Agreement", agr_display)

                st.caption(
                    "Confidence is the model's prediction score across sampled frames, "
                    "not an absolute guarantee of correctness."
                )

                # Review Required Alert
                st.markdown(
                    render_review_alert(
                        result["needs_review"],
                        result["decision_reason"],
                        float(result["confidence"]),
                        float(result["frame_agreement"]),
                    ),
                    unsafe_allow_html=True,
                )

                # Class probability breakdown
                st.markdown(
                    render_probability_bars(result.get("class_scores", {}), result.get("predicted_class", "")),
                    unsafe_allow_html=True,
                )

                # Agentic sampling metrics
                st.markdown(
                    f"""
                    <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 16px; margin: 16px 0;">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">
                            Agentic Vision Decision Details
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 0.9rem; color: #cbd5e1;">
                            <div>⏱ <strong>Inference Time:</strong> {inference_time:.1f} seconds</div>
                            <div>🎞 <strong>Frames Written:</strong> {result.get('frames_written', 0)} frames</div>
                            <div>🔄 <strong>Additional Sampling Pass:</strong> {'Yes (Uncertainty Triggered)' if result.get('retry_used') else 'No (Thresholds Satisfied in Pass 1)'}</div>
                            <div>💡 <strong>Decision Reason:</strong> {result.get('decision_reason', 'N/A')}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Annotated Video Preview & Download
                st.markdown(
                    """
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                        <span style="font-size: 1.2rem;">🎬</span>
                        <h4 style="margin: 0; color: #ffffff; font-size: 1.15rem;">
                            Annotated Video Output (YOLO11n Athlete Boxes & Prediction Banners)
                        </h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                vid_col_left, vid_col_center, vid_col_right = st.columns([1, 2, 1])
                with vid_col_center:
                    st.video(annotated_video_bytes, format="video/mp4")
                    st.download_button(
                        label="⬇  Download Annotated MP4",
                        data=annotated_video_bytes,
                        file_name="gymnalyze_annotated.mp4",
                        mime="video/mp4",
                        width="stretch",
                    )

            except subprocess.TimeoutExpired:
                st.error("Video conversion timed out. Please try a shorter video.")
            except FileNotFoundError as exc:
                st.error(f"Required file or utility not found: {exc}")
            except Exception as exc:
                st.error(f"Video analysis failed: {exc}")
            finally:
                for path in (temp_input, temp_output, temp_browser_output):
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass

# --------------------------------------------------
# SECTION 4: AGENTIC VISION EXPLANATION
# --------------------------------------------------

st.write("")
st.divider()
st.markdown(render_agentic_explanation(), unsafe_allow_html=True)
