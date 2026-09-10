import io
import time
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="NEON-NOIR // YOLO HUD",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# NEON NOIR STYLESHEET (CSS INJECTION)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Dark Noir Theme */
    :root {
        --bg-dark: #07090e;
        --card-bg: rgba(13, 17, 27, 0.75);
        --neon-cyan: #00f0ff;
        --neon-pink: #ff0055;
        --neon-purple: #9d00ff;
        --neon-amber: #ffaa00;
        --text-main: #e2e8f0;
        --text-dim: #798ba3;
    }

    .stApp {
        background: radial-gradient(circle at 50% 0%, #151128 0%, #08090f 60%, #030407 100%);
        color: var(--text-main);
        font-family: 'Rajdhani', sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase;
    }

    /* Cyber Title Glow */
    .cyber-title {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 2.2rem;
        color: #ffffff;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.7), 0 0 25px rgba(0, 240, 255, 0.4);
        margin-bottom: 0px;
    }

    .cyber-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: var(--neon-pink);
        letter-spacing: 2px;
        margin-bottom: 1.5rem;
    }

    /* HUD Metrics Cards */
    .hud-card {
        background: var(--card-bg);
        border: 1px solid rgba(0, 240, 255, 0.25);
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.08), inset 0 0 10px rgba(0, 240, 255, 0.03);
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 1rem;
        position: relative;
        backdrop-filter: blur(10px);
    }
    .hud-card::before {
        content: "";
        position: absolute;
        top: -1px; left: 10px; width: 40px; height: 2px;
        background: var(--neon-cyan);
        box-shadow: 0 0 8px var(--neon-cyan);
    }
    .hud-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .hud-val {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #fff;
    }
    .hud-val.cyan { color: var(--neon-cyan); text-shadow: 0 0 10px rgba(0,240,255,0.6); }
    .hud-val.pink { color: var(--neon-pink); text-shadow: 0 0 10px rgba(255,0,85,0.6); }
    .hud-val.amber { color: var(--neon-amber); text-shadow: 0 0 10px rgba(255,170,0,0.6); }

    /* Frame container for detection feed */
    .viewport-container {
        border: 1px solid rgba(255, 0, 85, 0.35);
        background: rgba(10, 10, 18, 0.85);
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 0 20px rgba(255, 0, 85, 0.12);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(7, 9, 14, 0.95);
        border-right: 1px solid rgba(0, 240, 255, 0.18);
    }
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--neon-cyan);
        font-size: 1.1rem !important;
    }

    /* Slider / Input Accents */
    div[data-baseweb="slider"] {
        filter: hue-rotate(130deg);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(0, 240, 255, 0.2);
        color: var(--text-dim);
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 240, 255, 0.1) !important;
        border-color: var(--neon-cyan) !important;
        color: var(--neon-cyan) !important;
    }

    /* Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(0, 240, 255, 0.15);
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# MODEL LOADER
# ---------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    try:
        from ultralytics import YOLO
        # yolov5su: upgraded YOLOv5s anchor-free model handled natively
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        return None

# ---------------------------------------------------------
# HEADER / BRANDING
# ---------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="cyber-title">SYNTEX // NEON-YOLO VISION</div>', unsafe_allow_html=True)
    st.markdown('<div class="cyber-subtitle">[SYS: ONLINE] // NEURAL OBJECT DETECTION ENGINE // REV 5su</div>', unsafe_allow_html=True)
with col_h2:
    st.markdown(
        """
        <div style="text-align: right; padding-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #00f0ff;">
            ⚡ TENSORFLOW / PYTORCH CORE<br>
            <span style="color:#ff0055;">● LIVE STREAM ACTIVE</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# MODEL INITIALIZATION
# ---------------------------------------------------------
with st.spinner("INITIATING NEURAL WEIGHTS..."):
    model = load_yolo_model()

if model is None:
    st.error("FATAL ERROR: Failed to mount neural model. Check requirements and internet connectivity.")
    st.stop()

# ---------------------------------------------------------
# HUD SIDEBAR CONFIGURATION
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### // SYSTEM TUNING")
    st.caption("Adjust confidence thresholding & IoU suppression parameters.")

    conf_threshold = st.slider(
        "CONFIDENCE THRESHOLD",
        min_value=0.05,
        max_value=1.0,
        value=0.30,
        step=0.01,
        help="Filters detections lower than this certainty."
    )

    iou_threshold = st.slider(
        "IOU THRESHOLD (NMS)",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Non-Maximum Suppression overlap threshold."
    )

    max_det = st.number_input(
        "MAX TARGETS (LIMIT)",
        min_value=1,
        max_value=500,
        value=50,
        step=5
    )

    st.markdown("---")
    st.markdown("### // TARGET FILTERING")
    
    # Allow filtering by specific classes
    all_class_names = list(model.names.values())
    selected_classes = st.multiselect(
        "TARGET CLASSES (EMPTY = ALL)",
        options=all_class_names,
        default=[]
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #506175;">
            NODE: YOLOv5su-ULTRALYTICS<br>
            PRECISION: FP32 / FP16 AUTO<br>
            SECURITY LEVEL: OVERRIDE
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# INPUT SOURCE SELECTION (CAM OR FILE)
# ---------------------------------------------------------
input_mode = st.radio(
    "SELECT INPUT STREAM",
    ["OPTICAL WEBCAM", "DIGITAL FILE UPLOAD"],
    horizontal=True,
    label_visibility="collapsed"
)

input_image = None

if input_mode == "OPTICAL WEBCAM":
    picture = st.camera_input("CAPTURE FEED", label_visibility="collapsed")
    if picture:
        input_image = Image.open(io.BytesIO(picture.getvalue())).convert("RGB")
else:
    uploaded_file = st.file_uploader(
        "DROP HIGH-RES IMAGE STREAM (JPG, PNG)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")

# ---------------------------------------------------------
# DETECTION ENGINE & TELEMETRY DASHBOARD
# ---------------------------------------------------------
if input_image is not None:
    # Filter class indices if selected
    class_filter_indices = None
    if selected_classes:
        class_filter_indices = [k for k, v in model.names.items() if v in selected_classes]

    # Model inference benchmark
    start_time = time.time()
    results = model(
        input_image,
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=int(max_det),
        classes=class_filter_indices
    )
    inference_time = (time.time() - start_time) * 1000

    res = results[0]
    boxes = res.boxes
    annotated_bgr = res.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]  # Ultralytics returns BGR; convert to RGB

    total_detections = len(boxes) if boxes is not None else 0
    top_confidence = float(boxes.conf.max().item()) if total_detections > 0 else 0.0
    detected_classes_count = len(set(boxes.cls.tolist())) if total_detections > 0 else 0

    # HUD KPI CARDS
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""<div class="hud-card">
                <div class="hud-label">// DETECTIONS</div>
                <div class="hud-val cyan">{total_detections}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with kpi2:
        st.markdown(
            f"""<div class="hud-card">
                <div class="hud-label">// UNIQUE CLASSES</div>
                <div class="hud-val pink">{detected_classes_count}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with kpi3:
        st.markdown(
            f"""<div class="hud-card">
                <div class="hud-label">// PEAK CONFIDENCE</div>
                <div class="hud-val amber">{top_confidence * 100:.1f}%</div>
            </div>""",
            unsafe_allow_html=True
        )
    with kpi4:
        st.markdown(
            f"""<div class="hud-card">
                <div class="hud-label">// INFERENCE LATENCY</div>
                <div class="hud-val">{inference_time:.1f} ms</div>
            </div>""",
            unsafe_allow_html=True
        )

    # VIEWPORT & TELEMETRY COLUMNS
    col_view, col_telemetry = st.columns([1.3, 1], gap="medium")

    with col_view:
        st.markdown("### // HUD VIEWPORT")
        view_tab1, view_tab2 = st.tabs(["[ ANNOTATED FEED ]", "[ RAW SENSOR FEED ]"])
        
        with view_tab1:
            st.image(
                annotated_rgb,
                caption="PROCESSED NEURAL MESH (YOLOv5su)",
                use_container_width=True
            )
        with view_tab2:
            st.image(
                input_image,
                caption="RAW UNFILTERED OPTIC INPUT",
                use_container_width=True
            )

    with col_telemetry:
        st.markdown("### // TELEMETRY & SPECTRUM")
        
        if total_detections > 0:
            category_count = {}
            category_conf = {}

            for box in boxes:
                c = int(box.cls.item())
                cf = float(box.conf.item())
