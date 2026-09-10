import io
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Object Detection Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# MOUSE-TRACKING SCRIPT (Passes coordinates to CSS)
# ---------------------------------------------------------
components.html(
    """
    <script>
    const parentDoc = window.parent.document;
    const root = parentDoc.documentElement;

    parentDoc.addEventListener('mousemove', (e) => {
        root.style.setProperty('--mouse-x', `${e.clientX}px`);
        root.style.setProperty('--mouse-y', `${e.clientY}px`);
    });
    </script>
    """,
    height=0,
    width=0
)

# ---------------------------------------------------------
# NEON NOIR DESIGN SYSTEM
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --bg-main: #090a0f;
        --card-bg: rgba(16, 18, 27, 0.75);
        --card-border: rgba(255, 255, 255, 0.08);
        --accent-cyan: #06b6d4;
        --accent-pink: #f43f5e;
        --accent-violet: #8b5cf6;
        --text-bright: #f8fafc;
        --text-muted: #94a3b8;
    }

    /* Interactive spotlight background responsive to cursor */
    .stApp {
        background-color: var(--bg-main);
        background-image: 
            radial-gradient(650px circle at var(--mouse-x, 50vw) var(--mouse-y, 30vh), rgba(6, 182, 212, 0.06), transparent 70%),
            radial-gradient(550px circle at calc(var(--mouse-x, 50vw) + 120px) calc(var(--mouse-y, 30vh) + 100px), rgba(244, 63, 94, 0.04), transparent 60%),
            linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px;
        color: var(--text-bright);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Typography */
    h1, h2, h3 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-bright) !important;
    }
    
    code, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Metric Card with Hover Glow */
    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 16px 20px;
        backdrop-filter: blur(12px);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        border-color: rgba(6, 182, 212, 0.45);
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(6, 182, 212, 0.15), 0 0 1px rgba(6, 182, 212, 0.5);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--text-bright);
    }

    /* Per-Item Detection Row / Card */
    .object-pill {
        background: rgba(22, 27, 38, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.2s ease;
    }
    .object-pill:hover {
        background: rgba(29, 36, 51, 0.9);
        border-color: rgba(244, 63, 94, 0.4);
        transform: translateX(3px);
        box-shadow: 0 4px 15px rgba(244, 63, 94, 0.1);
    }
    .object-name {
        font-weight: 600;
        font-size: 0.95rem;
        color: #f1f5f9;
        text-transform: capitalize;
    }
    .object-conf {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 5px;
        background: rgba(6, 182, 212, 0.12);
        color: var(--accent-cyan);
        border: 1px solid rgba(6, 182, 212, 0.25);
    }

    /* Panels & Containers */
    .panel-box {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(14px);
    }

    /* Sidebar Clean-up */
    section[data-testid="stSidebar"] {
        background-color: #0c0e15;
        border-right: 1px solid var(--card-border);
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
        return YOLO("yolov5su.pt")
    except Exception as e:
        return None

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    st.caption("Detection thresholds and filtering")

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.01,
        help="Minimum certainty score for a detection to be registered."
    )

    iou_threshold = st.slider(
        "IoU Overlap Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Non-Maximum Suppression threshold to merge overlapping boxes."
    )

    max_det = st.number_input(
        "Max Detections",
        min_value=1,
        max_value=500,
        value=100,
        step=10
    )

    model = load_yolo_model()

    if model:
        st.markdown("---")
        st.markdown("### Filter Classes")
        all_classes = sorted(list(model.names.values()))
        selected_classes = st.multiselect(
            "Target Classes",
            options=all_classes,
            default=[],
            help="Leave empty to detect all 80 COCO categories."
        )
    else:
        selected_classes = []

# ---------------------------------------------------------
# MAIN APP HEADER
# ---------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.title("Object Detection Studio")
    st.caption("YOLOv5su real-time visual inspection with multi-class telemetry")

with header_col2:
    input_source = st.segmented_control(
        "Input Mode",
        options=["Camera", "Upload"],
        default="Camera",
        label_visibility="collapsed"
    )

st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

if model is None:
    st.error("Failed to load YOLO model. Please verify your dependencies.")
    st.stop()

# ---------------------------------------------------------
# INPUT ACQUISITION
# ---------------------------------------------------------
input_image = None

if input_source == "Camera":
    camera_pic = st.camera_input("Capture frame", label_visibility="collapsed")
    if camera_pic:
        input_image = Image.open(io.BytesIO(camera_pic.getvalue())).convert("RGB")
else:
    uploaded = st.file_uploader(
        "Upload image (PNG, JPG, WebP)",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed"
    )
    if uploaded:
        input_image = Image.open(uploaded).convert("RGB")

# ---------------------------------------------------------
# PROCESSING & RESULTS WORKSPACE
# ---------------------------------------------------------
if input_image is not None:
    # Build class filter indices if active
    filter_indices = [k for k, v in model.names.items() if v in selected_classes] if selected_classes else None

    # Inference benchmark
    t_start = time.perf_counter()
    results = model(
        input_image,
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=int(max_det),
        classes=filter_indices
    )
    latency_ms = (time.perf_counter() - t_start) * 1000

    res = results[0]
    boxes = res.boxes
    annotated_rgb = res.plot()[:, :, ::-1]  # Ultralytics returns BGR; flip to RGB

    total_detections = len(boxes) if boxes is not None else 0
    unique_classes = len(set(boxes.cls.tolist())) if total_detections > 0 else 0
    top_conf = float(boxes.conf.max().item()) if total_detections > 0 else 0.0

    # Top Metric Bar
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Objects Found</div>
                <div class="metric-val" style="color: #06b6d4;">{total_detections}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Unique Categories</div>
                <div class="metric-val" style="color: #f43f5e;">{unique_classes}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Top Confidence</div>
                <div class="metric-val" style="color: #8b5cf6;">{top_conf * 100:.1f}%</div>
            </div>""",
            unsafe_allow_html=True
        )
    with m4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Latency</div>
                <div class="metric-val">{latency_ms:.1f}<span style="font-size: 1rem; color: #64748b;"> ms</span></div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)

    # Main Split-Screen Workspace
    viewport_col, data_col = st.columns([1.3, 1], gap="medium")

    with viewport_col:
        view_tabs = st.tabs(["Processed View", "Original View"])
        with view_tabs[0]:
            st.image(annotated_rgb, use_container_width=True)
        with view_tabs[1]:
            st.image(input_image, use_container_width=True)

    with data_col:
        st.markdown("### Object Inventory")

        if total_detections > 0:
            # Aggregate stats
            category_counts = {}
            category_confs = {}

            for box in boxes:
                c = int(box.cls.item())
                score = float(box.conf.item())
                category_counts[c] = category_counts.get(c, 0) + 1
                category_confs.setdefault(c, []).append(score)

            chart_data = pd.DataFrame({
                "Category": [model.names[c].capitalize() for c in category_counts.keys()],
                "Count": list(category_counts.values())
            }).sort_values(by="Count", ascending=False)

            # Frequency Chart
            st.bar_chart(chart_data.set_index("Category"), color="#06b6d4")

            # Interactive List of individual detected objects
            st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
            for box in sorted(boxes, key=lambda b: float(b.conf.item()), reverse=True)[:10]:
                cls_name = model.names[int(box.cls.item())].capitalize()
                conf_val = float(box.conf.item()) * 100
                st.markdown(
                    f"""<div class="object-pill">
                        <span class="object-name">{cls_name}</span>
                        <span class="object-conf">{conf_val:.1f}%</span>
                    </div>""",
                    unsafe_allow_html=True
                )

            if total_detections > 10:
                st.caption(f"Showing top 10 of {total_detections} detections.")

        else:
            st.info("No objects detected matching the current parameters. Lower the confidence threshold in the sidebar.")

else:
    # Clean placeholder when no image has been loaded
    st.markdown(
        """
        <div style="border: 1px dashed rgba(255,255,255,0.15); border-radius: 12px; padding: 4.5rem 1rem; text-align: center; margin-top: 1rem;">
            <div style="font-size: 1.15rem; font-weight: 600; color: #cbd5e1;">Awaiting image source</div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">Take a snapshot using the camera above or switch to file upload mode.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
