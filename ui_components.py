"""
ui_components.py — Reusable Modern UI Components & Styling for Gymnalyze.
Provides polished dark-mode aesthetics, metrics cards, probability bars,
and status badges for the hackathon demonstration dashboard.
"""

from typing import Dict, Any, Optional


def get_custom_css() -> str:
    """Return comprehensive CSS stylesheet for the Gymnalyze dark AI dashboard."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 50% 0%, #0d1527 0%, #080c16 100%) !important;
        color: #f1f5f9 !important;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 12, 22, 0.9) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .block-container {
        max-width: 1280px !important;
        padding-top: 2.2rem !important;
        padding-bottom: 4rem !important;
    }

    /* Gymnalyze Brand Navigation */
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0 24px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 24px;
    }

    .brand-title-wrap {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .brand-logo-icon {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.35);
    }

    .brand-name {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #ffffff;
        margin: 0;
        line-height: 1.1;
    }

    .brand-subtitle {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
        color: #94a3b8;
        text-transform: uppercase;
        margin-top: 3px;
    }

    /* Mode Selection Cards */
    .mode-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 24px;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
        cursor: pointer;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .mode-card:hover {
        border-color: #38bdf8;
        background: linear-gradient(180deg, #132238 0%, #0f172a 100%);
        transform: translateY(-4px);
        box-shadow: 0 14px 32px rgba(6, 182, 212, 0.2);
    }

    .mode-icon {
        font-size: 2.2rem;
        margin-bottom: 12px;
    }

    .mode-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .mode-desc {
        font-size: 0.9rem;
        color: #94a3b8;
        line-height: 1.55;
    }

    /* Status Pill Badges */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .status-pill.ready {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .status-pill.active {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    .status-pill.paused {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }

    .status-pill.processing {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
    }

    .status-pill.complete {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.5);
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: currentColor;
        box-shadow: 0 0 10px currentColor;
    }

    /* Results Dashboard Cards */
    .result-summary-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    }

    .result-stat-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s ease;
    }

    .result-stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
    }

    .result-stat-val {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
        margin: 4px 0;
    }

    .result-stat-val.highlight {
        color: #34d399;
    }

    .result-stat-lbl {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Dual Probability Bars */
    .prob-container {
        margin: 16px 0;
        background: #0b1120;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #1e293b;
    }

    .prob-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
        font-weight: 600;
        font-size: 0.95rem;
    }

    .prob-track {
        height: 12px;
        background: #1e293b;
        border-radius: 6px;
        overflow: hidden;
        position: relative;
        margin-bottom: 14px;
    }

    .prob-fill-artistic {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%);
        border-radius: 6px;
        transition: width 0.4s ease;
    }

    .prob-fill-rhythmic {
        height: 100%;
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        border-radius: 6px;
        transition: width 0.4s ease;
    }

    /* Agentic Vision Steps */
    .step-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 18px;
        height: 100%;
        transition: border-color 0.2s ease;
    }

    .step-card:hover {
        border-color: #38bdf8;
    }

    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 8px;
        background: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 10px;
    }

    .step-title {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }

    .step-desc {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    /* Warning / Review Needed Banner */
    .review-banner {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0;
        display: flex;
        align-items: center;
        gap: 14px;
        color: #fde68a;
    }

    .review-banner.verified {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #a7f3d0;
    }

    /* Button Tweaks */
    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.4rem !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button:hover {
        transform: translateY(-1px) !important;
    }

    div.stDownloadButton > button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.7rem 1.4rem !important;
    }

    /* Video player styling */
    video {
        border-radius: 14px !important;
        border: 1px solid #1e293b !important;
        max-height: 440px !important;
        width: 100% !important;
        background: #000000 !important;
    }

    /* Streamlit tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0b1120;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #1e293b;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 700;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
    }
    </style>
    """


def render_brand_header(subtitle: str = "Interactive AI Gymnastics Analysis Dashboard") -> str:
    """Render the standard Gymnalyze top branding banner."""
    return f"""
    <div class="brand-bar">
        <div class="brand-title-wrap">
            <div class="brand-logo-icon">🎯</div>
            <div>
                <h1 class="brand-name">GYMNALYZE</h1>
                <div class="brand-subtitle">{subtitle}</div>
            </div>
        </div>
        <div>
            <span class="status-pill ready">
                <span class="pulse-dot"></span>
                MobileNetV3 + YOLO11n Active
            </span>
        </div>
    </div>
    """


def render_status_pill(status_text: str, state: str = "ready") -> str:
    """
    Render a status badge.
    state can be: 'ready', 'active', 'paused', 'processing', 'complete'
    """
    return f"""
    <div style="margin: 10px 0;">
        <span class="status-pill {state}">
            <span class="pulse-dot"></span>
            {status_text}
        </span>
    </div>
    """


def render_probability_bars(class_scores: Dict[str, float], predicted_class: str) -> str:
    """Render a visual probability breakdown comparison between Artistic and Rhythmic."""
    artistic_score = float(class_scores.get("artistic", 0.0))
    rhythmic_score = float(class_scores.get("rhythmic", 0.0))

    artistic_pct = f"{artistic_score:.1%}"
    rhythmic_pct = f"{rhythmic_score:.1%}"

    artistic_width = max(0.0, min(100.0, artistic_score * 100))
    rhythmic_width = max(0.0, min(100.0, rhythmic_score * 100))

    is_none = (predicted_class or "").strip().lower() in ("none", "pending", "")
    artistic_badge = ' <span style="background: rgba(96,165,250,0.25); color: #93c5fd; font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; font-weight: 700; margin-left: 6px;">PREDICTED</span>' if (not is_none and predicted_class.lower() == "artistic") else ""
    rhythmic_badge = ' <span style="background: rgba(52,211,153,0.25); color: #6ee7b7; font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; font-weight: 700; margin-left: 6px;">PREDICTED</span>' if (not is_none and predicted_class.lower() == "rhythmic") else ""

    sub_text = "Awaiting live camera stream or video upload to calculate probabilities." if is_none else "Softmax probability distribution across both recognized gymnastics disciplines:"

    return (
        '<div class="prob-container">'
        '<div style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px;">Class Probability Distribution</div>'
        f'<div style="font-size: 0.78rem; color: #64748b; margin-bottom: 14px;">{sub_text}</div>'
        '<div class="prob-row">'
        f'<span style="color: #60a5fa;">🤸 Artistic Gymnastics{artistic_badge}</span>'
        f'<span style="font-family: \'JetBrains Mono\', monospace; color: #60a5fa; font-weight: 700;">{artistic_pct}</span>'
        '</div>'
        f'<div class="prob-track"><div class="prob-fill-artistic" style="width: {artistic_width:.1f}%;"></div></div>'
        '<div class="prob-row">'
        f'<span style="color: #34d399;">🎀 Rhythmic Gymnastics{rhythmic_badge}</span>'
        f'<span style="font-family: \'JetBrains Mono\', monospace; color: #34d399; font-weight: 700;">{rhythmic_pct}</span>'
        '</div>'
        f'<div class="prob-track"><div class="prob-fill-rhythmic" style="width: {rhythmic_width:.1f}%;"></div></div>'
        '</div>'
    )


def render_agentic_explanation() -> str:
    """Render the compact 6-step Agentic Vision explanation required by Section 4."""
    return """
    <div style="margin: 28px 0;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
            <span style="font-size: 1.4rem;">🧠</span>
            <h3 style="margin: 0; font-size: 1.35rem; font-weight: 700; color: #ffffff;">
                How Gymnalyze Analyses Gymnastics: Agentic Vision Architecture
            </h3>
        </div>
        <p style="color: #94a3b8; font-size: 0.92rem; line-height: 1.6; margin-bottom: 20px;">
            Gymnalyze combines lightweight deep learning with an adaptive multi-pass inspection loop.
            Unlike standard single-shot classifiers, it dynamically evaluates prediction uncertainty
            before committing to a final discipline judgment.
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px;">
            <div class="step-card">
                <div class="step-badge">1</div>
                <div class="step-title">Initial Frame Sampling</div>
                <div class="step-desc">
                    Extracts an initial set of evenly distributed frames across the video to capture apparatus transitions and athlete routines.
                </div>
            </div>
            <div class="step-card">
                <div class="step-badge">2</div>
                <div class="step-title">MobileNetV3 Classification</div>
                <div class="step-desc">
                    Each sampled frame passes through the trained MobileNetV3-Small neural network, producing normalized class probability vectors.
                </div>
            </div>
            <div class="step-card">
                <div class="step-badge">3</div>
                <div class="step-title">Confidence & Agreement Check</div>
                <div class="step-desc">
                    Evaluates mean prediction confidence against threshold (0.70) and measures frame-level consensus agreement (threshold 0.65).
                </div>
            </div>
            <div class="step-card">
                <div class="step-badge">4</div>
                <div class="step-title">Adaptive Frame Re-Sampling</div>
                <div class="step-desc">
                    If uncertainty thresholds are not met, the system autonomously triggers a secondary pass with 12 additional frames to resolve ambiguity.
                </div>
            </div>
            <div class="step-card">
                <div class="step-badge">5</div>
                <div class="step-title">YOLO11n Athlete Tracking</div>
                <div class="step-desc">
                    Ultralytics YOLO11n detects persons in every frame, extracting precise bounding coordinates and confidence scores for athletes.
                </div>
            </div>
            <div class="step-card">
                <div class="step-badge">6</div>
                <div class="step-title">Annotated Video Synthesis</div>
                <div class="step-desc">
                    Synthesizes bounding boxes, confidence badges, and prediction banners into a polished, downloadable H.264 MP4 video.
                </div>
            </div>
        </div>
        <div style="margin-top: 14px; padding: 12px 16px; background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; font-size: 0.82rem; color: #7dd3fc;">
            <strong>Pipeline Note:</strong> The 6-step frame-sampling and retry workflow applies to video files. The Live Camera mode streams continuous frames directly from your webcam, applying YOLO athlete detection and MobileNetV3 discipline scoring in real time.
        </div>
    </div>
    """


def render_review_alert(needs_review: bool, reason: str, confidence: float, agreement: Optional[float] = None) -> str:
    """Render a clean alert showing whether the prediction is confident or needs human review."""
    if needs_review:
        agree_text = f" | Frame Agreement: {agreement:.1%}" if agreement is not None else ""
        return f"""
        <div class="review-banner">
            <span style="font-size: 1.6rem;">⚠️</span>
            <div>
                <strong style="font-size: 0.95rem; color: #fbbf24; text-transform: uppercase; letter-spacing: 0.5px;">
                    Review Required — Uncertain Prediction
                </strong>
                <div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 2px;">
                    {reason} (Model Confidence: {confidence:.1%}{agree_text}). Human verification is recommended.
                </div>
            </div>
        </div>
        """
    else:
        agree_text = f" | Frame Agreement: {agreement:.1%}" if agreement is not None else ""
        return f"""
        <div class="review-banner verified">
            <span style="font-size: 1.6rem;">✅</span>
            <div>
                <strong style="font-size: 0.95rem; color: #34d399; text-transform: uppercase; letter-spacing: 0.5px;">
                    Prediction Verified
                </strong>
                <div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 2px;">
                    Prediction met all configured certainty thresholds (Model Confidence: {confidence:.1%}{agree_text}).
                </div>
            </div>
        </div>
        """
