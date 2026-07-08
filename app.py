"""
THE MAGNIFYING LENS - Detective Agency
A case-file styled object detection app.
Prepared by Er Ashish KC Khatri

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from pathlib import Path
import random
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from ultralytics import YOLO

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="The Magnifying Lens — Case File",
    page_icon="🔎",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent

# Noir palette
INK = "#0b0d0c"
PANEL = "#16181a"
PANEL_LIGHT = "#1e2124"
PAPER = "#e8dcc4"
BRASS = "#c9a84c"
CASE_RED = "#9c2b2b"
SLATE = "#7b8794"
GRAIN_LINE = "#2a2d2a"

# Verdict lines, chosen by detection count - flavor text, not analysis
VERDICT_LINES = {
    0: "No leads. The trail's gone cold.",
    1: "One suspect. Open and shut.",
    2: "Two persons of interest. Worth a second look.",
}
VERDICT_DEFAULT = "Multiple suspects in frame. This case has legs."

CLOSING_LINES = [
    "Filed under: solved, for now.",
    "The lens doesn't lie.",
    "Case logged at headquarters.",
    "Another one for the archive.",
]


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
def inject_noir_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Special+Elite&family=Oswald:wght@400;600;700&display=swap');

        .stApp {{
            background:
                radial-gradient(ellipse at 30% 0%, #1a1c1a 0%, {INK} 55%),
                repeating-linear-gradient(0deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 3px);
            color: {PAPER};
        }}

        h1, h2, h3, .case-heading {{
            font-family: 'Oswald', sans-serif;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: {PAPER};
        }}

        .agency-header {{
            border-bottom: 2px solid {BRASS};
            padding-bottom: 0.6rem;
            margin-bottom: 0.3rem;
        }}
        .agency-title {{
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            font-size: 2.2rem;
            letter-spacing: 0.06em;
            color: {PAPER};
            margin: 0;
        }}
        .agency-sub {{
            font-family: 'Special Elite', monospace;
            color: {BRASS};
            font-size: 0.95rem;
            margin-top: 0.15rem;
        }}

        .case-stamp {{
            display: inline-block;
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            letter-spacing: 0.12em;
            padding: 0.25rem 0.9rem;
            border: 3px solid {CASE_RED};
            color: {CASE_RED};
            transform: rotate(-3deg);
            border-radius: 4px;
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
        }}

        .evidence-tag {{
            background: {PANEL};
            border-left: 4px solid {BRASS};
            border-radius: 2px;
            padding: 0.7rem 1rem;
            margin-bottom: 0.6rem;
            font-family: 'Special Elite', monospace;
            position: relative;
        }}
        .evidence-tag .tag-number {{
            color: {BRASS};
            font-family: 'Oswald', sans-serif;
            font-weight: 600;
            letter-spacing: 0.08em;
            font-size: 0.78rem;
        }}
        .evidence-tag .tag-name {{
            color: {PAPER};
            font-size: 1.1rem;
            text-transform: capitalize;
            margin: 0.15rem 0;
        }}
        .evidence-tag .tag-certainty {{
            color: {SLATE};
            font-size: 0.85rem;
        }}
        .evidence-tag .tag-certainty.high {{ color: #b8d4a8; }}
        .evidence-tag .tag-certainty.low {{ color: #d4a8a8; }}

        .case-report {{
            background: {PANEL_LIGHT};
            border: 1px solid {GRAIN_LINE};
            border-radius: 3px;
            padding: 1.2rem 1.4rem;
            font-family: 'Special Elite', monospace;
        }}

        .verdict-line {{
            font-family: 'Special Elite', monospace;
            font-style: italic;
            color: {BRASS};
            font-size: 1.05rem;
            border-top: 1px dashed {GRAIN_LINE};
            padding-top: 0.6rem;
            margin-top: 0.8rem;
        }}

        [data-testid="stSidebar"] {{
            background: {PANEL};
            border-right: 2px solid {GRAIN_LINE};
        }}
        [data-testid="stSidebar"] * {{
            color: {PAPER} !important;
        }}

        .stButton>button, .stDownloadButton>button {{
            font-family: 'Oswald', sans-serif;
            letter-spacing: 0.05em;
            background: {BRASS};
            color: {INK};
            border: none;
            font-weight: 600;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background: {PANEL};
            border: 2px dashed {BRASS};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Briefing the investigator...")
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def get_available_weights() -> dict[str, str]:
    """Local .pt files next to app.py, plus standard presets.

    Note: legacy anchor-based yolov5s.pt checkpoints from the original
    ultralytics/yolov5 repo are not loadable via this package's YOLO()
    class - yolov5su.pt is the compatible rebuild if you need that family.
    """
    local_weights = {
        p.name: str(p) for p in APP_DIR.glob("*.pt") if p.name != "yolov5s.pt"
    }
    presets = {
        "YOLO11n — Field Agent (fastest, sharpest on CPU)": "yolo11n.pt",
        "YOLOv8n — Rookie Detective": "yolov8n.pt",
        "YOLOv8s — Seasoned Investigator": "yolov8s.pt",
        "YOLOv5su — Old-School Gumshoe": "yolov5su.pt",
        "YOLOv8m — Chief Inspector (slower, thorough)": "yolov8m.pt",
    }
    return {**local_weights, **presets}


# --------------------------------------------------------------------------
# Noir-style annotation
# --------------------------------------------------------------------------
def draw_case_file_boxes(image_rgb: np.ndarray, result) -> np.ndarray:
    """Draw brass evidence-tag style boxes instead of the default plot()."""
    canvas = image_rgb.copy()
    overlay = canvas.copy()

    brass_bgr = (76, 168, 201)  # BGR for #c9a84c-ish brass on screen
    red_bgr = (43, 43, 156)  # BGR for case red

    boxes = result.boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        name = result.names[cls_id]

        color = red_bgr if conf < 0.5 else brass_bgr

        # corner brackets instead of a full rectangle - evidence-marker feel
        bracket_len = max(12, int(min(x2 - x1, y2 - y1) * 0.18))
        thickness = 2
        corners = [
            ((x1, y1), (1, 1)),
            ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)),
            ((x2, y2), (-1, -1)),
        ]
        for (cx, cy), (dx, dy) in corners:
            cv2.line(canvas, (cx, cy), (cx + dx * bracket_len, cy), color, thickness)
            cv2.line(canvas, (cx, cy), (cx, cy + dy * bracket_len), color, thickness)

        label = f"#{i+1:02d}  {name.upper()}  {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 8, th + 4)
        cv2.rectangle(
            overlay, (x1, label_y - th - 6), (x1 + tw + 8, label_y + 4), (16, 16, 11), -1
        )
        canvas = cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0)
        cv2.putText(
            canvas, label, (x1 + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 1, cv2.LINE_AA,
        )

    # subtle vignette over the whole frame for noir mood
    h, w = canvas.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    dist = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    vignette = np.clip(1.0 - dist * 0.9, 0.55, 1.0)[..., None]
    canvas = (canvas.astype(np.float32) * vignette).clip(0, 255).astype(np.uint8)

    return canvas


def certainty_class(conf: float) -> str:
    if conf >= 0.7:
        return "high"
    if conf < 0.4:
        return "low"
    return ""


def certainty_word(conf: float) -> str:
    if conf >= 0.85:
        return "airtight"
    if conf >= 0.7:
        return "solid"
    if conf >= 0.5:
        return "plausible"
    if conf >= 0.3:
        return "shaky"
    return "a hunch, at best"


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
inject_noir_css()

st.sidebar.markdown("### 🗂 CASE SETTINGS")

weights_options = get_available_weights()
selected_label = st.sidebar.selectbox("Assign investigator", list(weights_options.keys()))
selected_weights = weights_options[selected_label]

confidence = st.sidebar.slider("Suspicion threshold", 0.05, 1.0, 0.25, 0.05)
iou = st.sidebar.slider("Overlap tolerance (IoU/NMS)", 0.05, 1.0, 0.45, 0.05)

st.sidebar.caption(
    "Local `.pt` files next to `app.py` are auto-enlisted. "
    "Preset investigators download once and are cached."
)

st.markdown(
    f"""
    <div class="agency-header">
        <p class="agency-title">🔎 The Magnifying Lens</p>
        <p class="agency-sub">Private Detection Agency — est. today, closes every case by inference</p>
    </div>
    <p style="color:{SLATE}; font-family:'Special Elite', monospace;">
        Prepared by Er Ashish KC Khatri, Lead Investigator
    </p>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Submit photographic evidence for the file (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    case_number = f"{random.randint(1000, 9999)}-{time.strftime('%y%m%d')}"
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown(
        f'<div class="case-stamp">CASE No. {case_number} — UNDER REVIEW</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="case-heading">Exhibit A — Submitted Photo</p>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    try:
        model = load_model(selected_weights)
    except Exception as e:
        st.error(
            f"Investigator failed to report for duty ('{selected_weights}'): {e}\n\n"
            "If this is happening on Streamlit Cloud, it's usually one of:\n"
            "- First-time download of preset weights failed (transient network issue on the host - try rerunning)\n"
            "- Out of memory on the free tier (try a leaner investigator, e.g. YOLOv8n)\n"
            "- Missing system libraries (check packages.txt includes libgl1)"
        )
        st.stop()

    with st.spinner("Dusting for prints..."):
        start = time.time()
        results = model.predict(
            source=np.array(image),
            conf=confidence,
            iou=iou,
            verbose=False,
        )
        elapsed = time.time() - start

    result = results[0]
    annotated_rgb = draw_case_file_boxes(np.array(image), result)

    with col2:
        st.markdown('<p class="case-heading">Exhibit B — Annotated Findings</p>', unsafe_allow_html=True)
        st.image(annotated_rgb, use_container_width=True)

    n = len(result.boxes)
    verdict = VERDICT_LINES.get(n, VERDICT_DEFAULT) if n > 0 else VERDICT_LINES[0]

    st.markdown('<p class="case-heading" style="margin-top:1.5rem;">📋 Case Report</p>', unsafe_allow_html=True)

    if n > 0:
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = result.names[cls_id]
            c_class = certainty_class(conf)
            c_word = certainty_word(conf)
            st.markdown(
                f"""
                <div class="evidence-tag">
                    <div class="tag-number">EVIDENCE #{i+1:02d}</div>
                    <div class="tag-name">{name}</div>
                    <div class="tag-certainty {c_class}">Certainty: {conf*100:.1f}% — {c_word}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"""
            <div class="case-report">
                No objects met the suspicion threshold. Either the scene is clean,
                or the threshold is set too high — try lowering it in the sidebar.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="case-report">
            Investigator: {selected_label}<br>
            Time on the case: {elapsed:.3f}s<br>
            Suspects identified: {n}
            <div class="verdict-line">"{verdict}" — {random.choice(CLOSING_LINES)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="case-report" style="margin-top:1rem;">
            The agency awaits its next case. Submit a photograph above to begin
            the investigation.
        </div>
        """,
        unsafe_allow_html=True,
    )
