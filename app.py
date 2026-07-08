"""
Object Detection Studio
Prepared by Er Ashish KC Khatri

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from pathlib import Path
import time

import numpy as np
import streamlit as st
from PIL import Image

# ultralytics wraps YOLOv8/v11 - no torch.hub / internet dependency needed
# for the model *code*, only the weights file matters.
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Object Detector",
    page_icon="🎯",
    layout="wide",
)

# Resolve paths relative to THIS FILE, not the current working directory.
# This is what fixes the "can Streamlit read yolov5s.pt" problem: relative
# paths break the moment you launch `streamlit run` from a different folder.
APP_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = APP_DIR / "yolov8n.pt"  # ultralytics auto-downloads this if missing


# --------------------------------------------------------------------------
# Model loading (cached properly)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(weights_path: str) -> YOLO:
    """
    Load a YOLO model once and reuse across reruns.

    IMPORTANT: use st.cache_resource (not st.cache_data) for models,
    connections, or anything that isn't a plain serializable value.
    cache_data tries to pickle the return value on every call, which
    either fails or silently wastes memory for large objects like models.
    """
    return YOLO(weights_path)


def get_available_weights() -> dict[str, str]:
    """Offer any .pt files found alongside the app, plus standard presets.

    NOTE on yolov5s.pt: checkpoints trained/downloaded from the original
    ultralytics/yolov5 repo (anchor-based) are NOT compatible with the
    `ultralytics` package's YOLO() loader. The fix is YOLOv5u - an
    anchor-free rebuild of the same architecture that Ultralytics
    maintains specifically for compatibility with this library. It's the
    same size class and same COCO classes, just loads correctly here.
    If you have a genuine legacy yolov5s.pt sitting in the repo, it will
    NOT work with this app - use yolov5su.pt below instead (auto-downloads
    on first run, then caches locally).
    """
    local_weights = {
        p.name: str(p) for p in APP_DIR.glob("*.pt") if p.name != "yolov5s.pt"
    }
    presets = {
        "YOLO11n (nano - fastest, best CPU accuracy/speed)": "yolo11n.pt",
        "YOLOv8n (nano)": "yolov8n.pt",
        "YOLOv8s (small - balanced)": "yolov8s.pt",
        "YOLOv5su (small - legacy-compatible)": "yolov5su.pt",
        "YOLOv8m (medium - more accurate)": "yolov8m.pt",
    }
    # local custom weights take priority in the list
    return {**local_weights, **presets}


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("Settings")

weights_options = get_available_weights()
selected_label = st.sidebar.selectbox("Model", list(weights_options.keys()))
selected_weights = weights_options[selected_label]

confidence = st.sidebar.slider("Confidence threshold", 0.05, 1.0, 0.25, 0.05)
iou = st.sidebar.slider("IoU threshold (NMS)", 0.05, 1.0, 0.45, 0.05)

st.sidebar.caption(
    "Local `.pt` files placed next to `app.py` show up automatically. "
    "Preset names (e.g. `yolov8n.pt`) download once and are cached by ultralytics."
)

# --------------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------------
st.title("🎯 Object Detection")
st.write("**Prepared by Er Ashish KC Khatri**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    try:
        model = load_model(selected_weights)
    except Exception as e:
        st.error(
            f"Could not load model weights '{selected_weights}': {e}\n\n"
            "If this is happening on Streamlit Cloud, it's usually one of:\n"
            "- First-time download of preset weights failed (transient network issue on the host - try rerunning)\n"
            "- Out of memory on the free tier (try a smaller model like yolov8n.pt)\n"
            "- Missing system libraries (check packages.txt includes libgl1)"
        )
        st.stop()

    with st.spinner("Running detection..."):
        start = time.time()
        results = model.predict(
            source=np.array(image),
            conf=confidence,
            iou=iou,
            verbose=False,
        )
        elapsed = time.time() - start

    result = results[0]
    annotated = result.plot()  # BGR numpy array with boxes/labels drawn in
    annotated_rgb = annotated[:, :, ::-1]  # BGR -> RGB for st.image

    with col2:
        st.image(annotated_rgb, caption="Detected Objects", use_container_width=True)

    st.caption(f"Inference time: {elapsed:.3f}s on {len(result.boxes)} detection(s)")

    # Structured results table instead of a flat text list
    if len(result.boxes) > 0:
        rows = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            rows.append(
                {
                    "class": result.names[cls_id],
                    "confidence": float(box.conf[0]),
                    "x1": float(box.xyxy[0][0]),
                    "y1": float(box.xyxy[0][1]),
                    "x2": float(box.xyxy[0][2]),
                    "y2": float(box.xyxy[0][3]),
                }
            )
        st.subheader("Detections")
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No objects detected above the confidence threshold.")
else:
    st.info("Upload an image to get started.")
