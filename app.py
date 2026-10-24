"""
app.py — Gymnalyze Landing Dashboard & Hackathon Showcase.
Presents the AI gymnastics analysis platform, architecture explanation,
disciplines showcase, and direct entry points to both testing modes:
  Mode A: Live Camera Analysis
  Mode B: Upload Video Analysis
"""

from pathlib import Path
import streamlit as st
import ui_components
from ui_components import (
    get_custom_css,
    render_brand_header,
    render_status_pill,
    render_agentic_explanation,
)

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

st.set_page_config(
    page_title="Gymnalyze | AI Gymnastics Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(get_custom_css(), unsafe_allow_html=True)


def get_asset_image(filename_base: str) -> Path:
    """Return path to asset image checking both .jpeg and .jpg extensions."""
    for ext in (".jpeg", ".jpg", ".png"):
        p = ASSETS_DIR / f"{filename_base}{ext}"
        if p.is_file():
            return p
    return ASSETS_DIR / f"{filename_base}.jpeg"


# --------------------------------------------------
# TOP NAVIGATION & BRAND
# --------------------------------------------------
st.markdown(
    render_brand_header(subtitle="Hackathon Showcase Edition · Real-Time Edge AI & Agentic Vision"),
    unsafe_allow_html=True,
)

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------
hero_left, hero_right = st.columns([1.15, 1], gap="large")

with hero_left:
    st.markdown(
        """
        <div style="display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 9999px; font-size: 0.8rem; font-weight: 700; color: #34d399; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 14px;">
            ⚡ AI-POWERED SPORTS ANALYTICS
        </div>
        <h1 style="font-size: clamp(2.6rem, 5vw, 3.8rem); line-height: 1.1; font-weight: 850; color: #ffffff; margin: 0 0 16px 0;">
            Decode Every<br>
            <span style="background: linear-gradient(90deg, #34d399 0%, #38bdf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Gymnastics Move.
            </span>
        </h1>
        <p style="font-size: 1.05rem; line-height: 1.7; color: #cbd5e1; max-width: 580px; margin-bottom: 24px;">
            Gymnalyze revolutionizes gymnastics video analytics by pairing <strong>MobileNetV3-Small</strong> deep classification with <strong>YOLO11n</strong> real-time athlete tracking and an adaptive multi-pass <strong>Agentic Vision</strong> loop.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Quick Jump Buttons
    cta_col1, cta_col2 = st.columns(2, gap="small")
    with cta_col1:
        if st.button("🎥 Launch Live Camera Mode", type="primary", width="stretch"):
            st.session_state["selected_mode"] = "live"
            st.switch_page("pages/2_Video_Analysis.py")
    with cta_col2:
        if st.button("📁 Launch Video Upload Mode", width="stretch"):
            st.session_state["selected_mode"] = "upload"
            st.switch_page("pages/2_Video_Analysis.py")

    st.caption("Supports live webcam stream analysis and video file uploads (MP4, MOV, AVI).")

with hero_right:
    hero_img = get_asset_image("hero_gymnastics")
    if hero_img.is_file():
        st.image(str(hero_img), width="stretch")
    else:
        st.info("Hero visual ready.")

# --------------------------------------------------
# TWO PROMINENT MODE SELECTION CARDS
# --------------------------------------------------
st.write("")
st.divider()

st.markdown(
    """
    <div style="text-align: center; margin-bottom: 24px;">
        <h2 style="font-size: 1.85rem; font-weight: 800; color: #ffffff; margin-bottom: 6px;">
            Choose Your Analysis Mode
        </h2>
        <p style="color: #94a3b8; font-size: 0.95rem; max-width: 620px; margin: 0 auto;">
            Experience Gymnalyze through live webcam motion tracking or deep multi-pass video file evaluation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

mode_c1, mode_c2 = st.columns(2, gap="large")

with mode_c1:
    st.markdown(
        """
        <div class="mode-card">
            <div>
                <div class="mode-icon">🎥</div>
                <div class="mode-title">Mode A — Live Camera Analysis</div>
                <div class="mode-desc">
                    Connect your webcam to perform real-time athlete detection, live discipline probability tracking, and instant annotated MP4 session recording.
                </div>
                <div style="margin: 16px 0; display: flex; flex-wrap: wrap; gap: 8px;">
                    <span style="font-size: 0.75rem; background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 4px 10px; border-radius: 6px; font-weight: 600;">Hardware Webcam</span>
                    <span style="font-size: 0.75rem; background: rgba(16, 185, 129, 0.15); color: #34d399; padding: 4px 10px; border-radius: 6px; font-weight: 600;">Live Bounding Boxes</span>
                    <span style="font-size: 0.75rem; background: rgba(168, 85, 247, 0.15); color: #c084fc; padding: 4px 10px; border-radius: 6px; font-weight: 600;">Instant Recording</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Start Live Camera Analysis →", key="btn_mode_a", width="stretch"):
        st.session_state["selected_mode"] = "live"
        st.switch_page("pages/2_Video_Analysis.py")

with mode_c2:
    st.markdown(
        """
        <div class="mode-card">
            <div>
                <div class="mode-icon">📁</div>
                <div class="mode-title">Mode B — Upload Video Analysis</div>
                <div class="mode-desc">
                    Upload pre-recorded competition routines. Features multi-pass adaptive frame sampling, uncertainty threshold checks, and full annotated video export.
                </div>
                <div style="margin: 16px 0; display: flex; flex-wrap: wrap; gap: 8px;">
                    <span style="font-size: 0.75rem; background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 4px 10px; border-radius: 6px; font-weight: 600;">Drag & Drop Video</span>
                    <span style="font-size: 0.75rem; background: rgba(16, 185, 129, 0.15); color: #34d399; padding: 4px 10px; border-radius: 6px; font-weight: 600;">Adaptive Sampling</span>
                    <span style="font-size: 0.75rem; background: rgba(168, 85, 247, 0.15); color: #c084fc; padding: 4px 10px; border-radius: 6px; font-weight: 600;">Download Annotated MP4</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Start Video Upload Analysis →", key="btn_mode_b", width="stretch"):
        st.session_state["selected_mode"] = "upload"
        st.switch_page("pages/2_Video_Analysis.py")

# --------------------------------------------------
# SECTION 4: AGENTIC VISION EXPLANATION
# --------------------------------------------------
st.write("")
st.divider()
st.markdown(render_agentic_explanation(), unsafe_allow_html=True)

# --------------------------------------------------
# DISCIPLINES SHOWCASE
# --------------------------------------------------
st.write("")
st.divider()

st.markdown(
    """
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 1.7rem; font-weight: 800; color: #ffffff; margin-bottom: 6px;">
            Recognized Gymnastics Disciplines
        </h2>
        <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">
            The classification backbone is trained to distinguish between the two primary Olympic gymnastics categories:
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

disc_c1, disc_c2 = st.columns(2, gap="large")

with disc_c1:
    art_img = get_asset_image("artistic_gymnastics")
    if art_img.is_file():
        st.image(str(art_img), width="stretch")
    st.markdown(
        """
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 18px; margin-top: 10px;">
            <h3 style="color: #60a5fa; margin: 0 0 6px 0; font-size: 1.2rem;">🤸 Artistic Gymnastics</h3>
            <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6; margin: 0;">
                Characterized by explosive acrobatics, apparatus work (vault, uneven bars, balance beam, floor, rings, pommel horse), and dynamic rotational dismounts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with disc_c2:
    rhy_img = get_asset_image("rhythmic_gymnastics")
    if rhy_img.is_file():
        st.image(str(rhy_img), width="stretch")
    st.markdown(
        """
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 18px; margin-top: 10px;">
            <h3 style="color: #34d399; margin: 0 0 6px 0; font-size: 1.2rem;">🎀 Rhythmic Gymnastics</h3>
            <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6; margin: 0;">
                Focuses on expressive musical choreography, extreme body flexibility, and graceful manipulation of handheld apparatus including ribbons, hoops, balls, and clubs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# FOOTER / HACKATHON CREDITS
# --------------------------------------------------
st.write("")
st.divider()

st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; font-size: 0.85rem; color: #64748b; padding: 12px 0;">
        <div>
            🎯 <strong>GYMNALYZE</strong> · AI Gymnastics Video & Pose Analysis Dashboard
        </div>
        <div>
            Powered by PyTorch · TorchVision · Ultralytics YOLO11n · OpenCV 5.0
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
