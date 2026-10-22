
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

st.set_page_config(
    page_title="Gymnalyze | AI Gymnastics Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .stApp {
        background: #0b1020;
        color: #f8fafc;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.95);
    }

    .block-container {
        max-width: 1200px;
        padding-top: 4rem !important;
        padding-bottom: 3rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        font-family: "Trebuchet MS", Arial, sans-serif;
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: 2px;
        line-height: 1.4;
        color: #ffffff;
        white-space: nowrap;
        width: 100%;
        overflow: visible;
        padding: 8px 0;
        position: relative;
        z-index: 1;
        display: flex;
        padding-top: 12px;
        padding-bottom: 12px;
    }
    
    .brand-icon {
        font-size: 1.6rem;
        letter-spacing: 0;
        flex-shrink: 0;
    }

    .eyebrow {
        color: #4ade80;
        font-weight: 700;
        letter-spacing: 2px;
        font-size: 0.8rem;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: clamp(2.5rem, 5vw, 4.3rem);
        line-height: 1.08;
        font-weight: 850;
        color: #ffffff;
        margin: 0.7rem 0 1rem 0;
    }

    .hero-description {
        color: #cbd5e1;
        font-size: 1.1rem;
        line-height: 1.8;
    }

    .section-title {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 750;
        margin-top: 1rem;
    }

    .feature-card {
        background: #151d30;
        border: 1px solid #29364d;
        border-radius: 16px;
        padding: 22px;
        min-height: 165px;
        margin-bottom: 12px;
    }

    .feature-card h3 {
        color: #ffffff;
        font-size: 1.15rem;
    }

    .feature-card p {
        color: #b9c5d8;
        line-height: 1.65;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 48px;
    }

    hr {
        border-color: #29364d;
    }
    
    [data-testid="stSidebar"] {
             display: none;
            }
    
    [data-testid="collapsedControl"] {
            display: none;
            }
            
    .hero-title {
        font-family: "Trebuchet MS", sans-serif;
        font-size: clamp(2.8rem, 5vw, 4.5rem);
        font-weight: 800;
        line-height: 1.08;
        letter-spacing: -1.5px;
        color: #ffffff;
        margin: 1rem 0;
    }

    .hero-description {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #cbd5e1;
        max-width: 540px; 
    }
    
    /* Keep the default Streamlit header visible */
    [data-testid="stHeader"] {
        height: 3.5rem;
        background: #0b1020;
    }

    [data-testid="stImage"] img {
        border-radius: 18px;
    }
 
    div.stButton > button[kind="primary"] {
        min-height: 52px;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 700;
    }
    [data-testid="stImage"] img {
        width: 100%;
        border-radius: 18px;
        object-fit: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Navigation / brand ----------
header_left, header_right = st.columns([3, 1], gap="small")

with header_left:
    st.markdown(
        """
        <div class="brand">
            <span class="brand-icon">🎯</span>
            <span>GYMNALYZE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    if st.button(
        "Try Video Analysis →",
        type="primary",
        width="stretch",
        key="header_analysis",
    ):
        st.switch_page("pages/2_Video_Analysis.py")

# ---------- Hero section ----------
hero_left, hero_right = st.columns([1.1, 1], gap="large")

with hero_left:
    st.markdown(
        '<div class="eyebrow">AI-POWERED SPORTS ANALYTICS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-title">
            Decode Every<br>
            <span style="color:#4ade80;">Gymnastics Move.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-description">
            Discover Gymnalyze — a computer-vision project that
            classifies gymnastics videos, detects athletes,
            and generates annotated results.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    if st.button(
        "▶  Start Video Analysis",
        type="primary",
        width="stretch",
    ):
        st.switch_page("pages/2_Video_Analysis.py")

    st.caption("Upload a video. Explore predictions. Download results.")

with hero_right:
    hero_image = ASSETS_DIR / "hero_gymnastics.jpeg"

    if hero_image.is_file():
        st.image(
            str(hero_image),
            width="stretch",
        )
    else:
        st.error(f"Hero image not found: {hero_image}")

# ---------- About section ----------
st.write("")
st.divider()

st.markdown(
    '<div class="section-title">What is Gymnalyze?</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    Gymnalyze explores how computer vision and deep learning
    can help analyze gymnastics videos. Its video-analysis
    pipeline predicts a gymnastics discipline, applies athlete
    detection, displays the prediction score, and creates an
    annotated video that can be downloaded.
    """
)

st.write("")

# ---------- Features ----------
st.markdown(
    '<div class="section-title">What can it do?</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3, gap="medium")

features = [
    (
        "🧠",
        "AI Classification",
        "Use the trained MobileNetV3 model to predict the "
        "discipline represented in a video.",
    ),
    (
        "🔎",
        "Athlete Detection",
        "Use YOLO-based person detection to identify athletes "
        "and visualize detection boxes.",
    ),
    (
        "🎬",
        "Annotated Output",
        "Review the processed video, see the prediction score, "
        "and download an MP4 result.",
    ),
]

for column, (emoji, title, description) in zip(
    [c1, c2, c3], features
):
    with column:
        st.markdown(
            f"""
            <div class="feature-card">
                <div style="font-size: 2rem;">{emoji}</div>
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------- Disciplines ----------
st.write("")
st.divider()

st.markdown(
    '<div class="section-title">Explore Gymnastics</div>',
    unsafe_allow_html=True,
)

st.caption(
    "The available classification categories depend on the "
    "classes included in the trained model."
)

d1, d2 = st.columns(2, gap="large")

with d1:
    artistic = ASSETS_DIR / "artistic_gymnastics.jpg"

    if artistic.exists():
        st.image(str(artistic), use_container_width=True)

    st.subheader("Artistic Gymnastics")
    st.write(
        "A discipline featuring strength, balance, flexibility, "
        "and acrobatic skills performed on apparatus or the floor."
    )

with d2:
    rhythmic = ASSETS_DIR / "rhythmic_gymnastics.jpg"

    if rhythmic.exists():
        st.image(str(rhythmic), use_container_width=True)

    st.subheader("Rhythmic Gymnastics")
    st.write(
        "A discipline combining expressive body movements, "
        "coordination, and apparatus such as ribbons or hoops."
    )

# ---------- Call to action ----------
st.write("")
st.divider()

st.markdown(
    '<div class="section-title">Ready to try Gymnalyze?</div>',
    unsafe_allow_html=True,
)

st.write(
    "Upload an unseen gymnastics video and explore the "
    "model's prediction and annotated output."
)

if st.button(
    "Open Video Analysis →",
    type="primary",
    use_container_width=True,
):
    st.switch_page("pages/2_Video_Analysis.py")

st.write("")
st.markdown(
    """
    <div class="small-note">
        GYMNALYZE · AI-powered gymnastics video analysis ·
        Academic / experimental project
    </div>
    """,
    unsafe_allow_html=True,
)
