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
    page_title="Object Detection Studio // YOLOv5",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# MOUSE TRACKER & AMBIENT AURA SCRIPT
# ---------------------------------------------------------
components.html(
    """
    <script>
    const pDoc = window.parent.document;
    const root = pDoc.documentElement;

    // Create luminous cursor follower element
    let aura = pDoc.getElementById('neon-mouse-aura');
    if (!aura) {
        aura = pDoc.createElement('div');
        aura.id = 'neon-mouse-aura';
        aura.style.position = 'fixed';
        aura.style.pointerEvents = 'none';
        aura.style.width = '380px';
        aura.style.height = '380px';
        aura.style.borderRadius = '50%';
        aura.style.background = 'radial-gradient(circle, rgba(255, 0, 128, 0.18) 0%, rgba(0, 240, 255, 0.12) 40%, transparent 70%)';
        aura.style.transform = 'translate(-50%, -50%)';
        aura.style.transition = 'transform 0.05s ease-out, opacity 0.2s ease';
        aura.style.zIndex = '999999';
        aura.style.mixBlendMode = 'screen';
        aura.style.filter = 'blur(12px)';
        pDoc.body.appendChild(aura);
    }

    pDoc.addEventListener('mousemove', (e) => {
        root.style.setProperty('--mouse-x', `${e.clientX}px`);
        root.style.setProperty('--mouse-y', `${e.clientY}px`);
        aura.style.left = `${e.clientX}px`;
        aura.style.top = `${e.clientY}px`;
    });
    </script>
    """,
    height=0,
    width=0
)

# ---------------------------------------------------------
# OVER-THE-TOP NEON NOIR STYLESHEET
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@400;600;800;900&family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@500;700&display=swap');

    :root {
        --neon-cyan: #00f0ff;
        --neon-pink: #ff007f;
        --neon-violet: #9d00ff;
        --neon-amber: #ffb703;
        --dark-void: #05060a;
        --card-surface: rgba(13, 16, 28, 0.85);
    }

    /* Interactive cursor-following spotlight canvas */
    .stApp {
        background-color: var(--dark-void);
        background-image: 
            radial-gradient(750px circle at var(--mouse-x, 50vw) var(--mouse-y, 30vh), rgba(0, 240, 255, 0.12), transparent 60%),
            radial-gradient(600px circle at calc(var(--mouse-x, 50vw) - 180px) calc(var(--mouse-y, 30vh) + 120px), rgba(255, 0, 127, 0.10), transparent 55%),
            linear-gradient(rgba(0, 240, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 0, 127, 0.03) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 42px 42px, 42px 42px;
        color: #f8fafc;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Animated Multi-Gradient Title */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .ultra-title {
        font-family: 'Unbounded', sans-serif;
        font-weight: 900;
        font-size: 2.35rem;
        letter-spacing: -0.04em;
        background: linear-gradient(90deg, #00f0ff, #ff007f, #9d00ff, #00f0ff);
        background-size: 300% 300%;
        animation: gradientShift 6s linear infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 35px rgba(255, 0, 127, 0.4);
        margin: 0;
    }

    .sub-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        color: var(--neon-cyan);
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-top: 4px;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        background-color: var(--neon-pink);
        border-radius: 50%;
        box-shadow: 0 0 10px var(--neon-pink), 0 0 20px var(--neon-pink);
        animation: pulseDot 1.5s infinite;
    }

    @keyframes pulseDot {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.4); opacity: 0.5; }
    }

    /* Top Control Deck Strip */
    .control-deck {
        background: var(--card-surface);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 14px;
        padding: 16px 22px;
        box-shadow: 0 0 30px rgba(0, 240, 255, 0.08), inset 0 0 15px rgba(0, 240, 255, 0.03);
        backdrop-filter: blur(16px);
        margin-bottom: 1.5rem;
    }

    /* Saturated KPI Cards */
    .kpi-card {
        background: var(--card-surface);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 12px;
        padding: 16px 20px;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, var(--neon-cyan), var(--neon-pink));
        opacity: 0.5;
        transition: opacity 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px) scale(1.01);
        border-color: var(--neon-cyan);
        box-shadow: 0 12px 30px rgba(0, 240, 255, 0.2), 0 0 15px rgba(255, 0, 127, 0.2);
    }
    .kpi-card:hover::after {
        opacity: 1;
    }
    .kpi-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
    }
    .kpi-num {
        font-family: 'Unbounded', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-top: 4px;
    }

    /* Neon Viewport Container with Cyber Brackets */
    .viewport-frame {
        position: relative;
        background: rgba(10, 13, 22, 0.9);
        border: 1px solid rgba(0, 240, 255, 0.3);
        border-radius: 14px;
        padding: 12px;
        box-shadow: 0 0 35px rgba(0, 240, 255, 0.12), inset 0 0 30px rgba(0, 0, 0, 0.6);
        overflow: hidden;
    }
    .viewport-frame::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, var(--neon-cyan), var(--neon-pink), transparent);
        animation: scanline 4s linear infinite;
    }
    @keyframes scanline {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(1200%); }
    }

    /* Detection Item Card with Gradient Confidence Meter */
    .det-item {
        background: rgba(18, 22, 36, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .det-item:hover {
        background: rgba(26, 31, 51, 0.95);
        border-color: var(--neon-pink);
        transform: translateX(5px);
        box-shadow: 0 0 20px rgba(255, 0, 127, 0.25);
    }
    .det-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .det-name {
        font-family: 'Unbounded', sans-serif;
        font-weight: 700;
        font-size: 0.88rem;
        letter-spacing: -0.02em;
        color: #ffffff;
        text-transform: capitalize;
    }
    .det-conf-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--neon-cyan);
    }
    .det-meter-bg {
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        overflow: hidden;
    }
    .det-meter-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--neon-cyan), var(--neon-violet), var(--neon-pink));
        border-radius: 999px;
        box-shadow: 0 0 10px rgba(255, 0, 127, 0.6);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# MODEL LOADER
# ---------------------------------------------------------
@st.cache_resource
def load_yolo():
    try:
        from ultralytics import YOLO
        return YOLO("yolov5su.pt")
    except Exception as e:
        return None

model = load_yolo()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.markdown('<div class="ultra-title">OBJECT DETECTION STUDIO</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-badge"><span class="live-dot"></span> YOLOv5su ENGINE ACTIVE • INTERACTIVE VIEWPORT</div>',
        unsafe_allow_html=True
    )

with head_col2:
    input_source = st.radio(
        "Source",
        ["Live Camera", "Image Upload"],
        horizontal=True,
        label_visibility="collapsed"
    )

st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)

if model is None:
    st.error("Failed to mount YOLOv5 model. Check dependencies.")
    st.stop()

# ---------------------------------------------------------
# INTEGRATED CONTROL DECK
# ---------------------------------------------------------
with st.expander("⚡ DETECTION PARAMETERS & FILTERS", expanded=True):
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 2])

    with ctrl_col1:
        conf_thresh = st.slider("Confidence Threshold", 0.05, 1.0, 0.28, 0.01)

    with ctrl_col2:
        iou_thresh = st.slider("IoU Overlap Suppression", 0.10, 1.0, 0.45, 0.05)

    with ctrl_col3:
        all_coco = sorted(list(model.names.values()))
        selected_classes = st.multiselect(
            "Filter Target Classes",
            options=all_coco,
            default=[],
            placeholder="All 80 classes enabled"
        )

# ---------------------------------------------------------
# ACQUIRE INPUT FRAME
# ---------------------------------------------------------
input_image = None

if input_source == "Live Camera":
    shot = st.camera_input("Optical Feed Capture", label_visibility="collapsed")
    if shot:
        input_image = Image.open(io.BytesIO(shot.getvalue())).convert("RGB")
else:
    up = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if up:
        input_image = Image.open(up).convert("RGB")

# ---------------------------------------------------------
# INFERENCE & MULTI-PANE WORKSPACE
# ---------------------------------------------------------
if input_image is not None:
    filter_ids = [k for k, v in model.names.items() if v in selected_classes] if selected_classes else None

    # Benchmark inference
    t0 = time.perf_counter()
    results = model(
        input_image,
        conf=conf_thresh,
        iou=iou_thresh,
        classes=filter_ids
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    res = results[0]
    boxes = res.boxes
    annotated_rgb = res.plot()[:, :, ::-1]

    count_total = len(boxes) if boxes is not None else 0
    unique_types = len(set(boxes.cls.tolist())) if count_total > 0 else 0
    peak_conf = float(boxes.conf.max().item()) * 100 if count_total > 0 else 0.0

    # 4 HIGH-CONTRAST METRICS
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-title">Detected Objects</div>
                <div class="kpi-num" style="color: var(--neon-cyan);">{count_total}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-title">Unique Classes</div>
                <div class="kpi-num" style="color: var(--neon-pink);">{unique_types}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-title">Top Confidence</div>
                <div class="kpi-num" style="color: #00ff88;">{peak_conf:.1f}%</div>
            </div>""",
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-title">Speed Latency</div>
                <div class="kpi-num" style="color: #ffb703;">{inference_ms:.1f}<span style="font-size: 1rem; color: #64748b;">ms</span></div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

    # 2-PANE WORKSPACE: VIEWPORT (LEFT) vs TELEMETRY TOWER (RIGHT)
    col_viewport, col_telemetry = st.columns([1.35, 1], gap="large")

    with col_viewport:
        view_mode = st.segmented_control(
            "Channel",
            options=["Annotated Overlay", "Raw Sensor View"],
            default="Annotated Overlay",
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="viewport-frame">', unsafe_allow_html=True)
        if view_mode == "Annotated Overlay":
            st.image(annotated_rgb, use_container_width=True)
        else:
            st.image(input_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_telemetry:
        st.markdown("<h3 style='font-family: Unbounded; font-size: 1.1rem; margin-top: 0;'>OBJECT INVENTORY</h3>", unsafe_allow_html=True)

        if count_total > 0:
            # Category Breakdown
            cat_counts = {}
            for b in boxes:
                c = int(b.cls.item())
                cat_counts[c] = cat_counts.get(c, 0) + 1

            df_chart = pd.DataFrame({
                "Category": [model.names[c].capitalize() for c in cat_counts.keys()],
                "Count": list(cat_counts.values())
            }).sort_values(by="Count", ascending=False)

            # Saturated bar chart
            st.bar_chart(df_chart.set_index("Category"), color="#00f0ff")

            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

            # Individual Detection Pills with animated gradient progress bars
            for b in sorted(boxes, key=lambda x: float(x.conf.item()), reverse=True)[:8]:
                cls_label = model.names[int(b.cls.item())].capitalize()
                c_score = float(b.conf.item()) * 100
                st.markdown(
                    f"""<div class="det-item">
                        <div class="det-header">
                            <span class="det-name">{cls_label}</span>
                            <span class="det-conf-val">{c_score:.1f}%</span>
                        </div>
                        <div class="det-meter-bg">
                            <div class="det-meter-fill" style="width: {c_score}%;"></div>
                        </div>
                    </div>""",
                    unsafe_allow_html=True
                )

            if count_total > 8:
                st.caption(f"Displaying top 8 of {count_total} detected instances.")
        else:
            st.markdown(
                """
                <div style="border: 1px solid rgba(255, 0, 127, 0.4); border-radius: 12px; padding: 2.5rem 1rem; text-align: center; background: rgba(255, 0, 127, 0.05);">
                    <div style="font-family: Unbounded; font-size: 1rem; color: var(--neon-pink); margin-bottom: 6px;">Zero Objects Detected</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">Adjust the confidence threshold slider above or check your class filters.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

else:
    st.markdown(
        """
        <div style="border: 1px dashed rgba(0, 240, 255, 0.3); border-radius: 16px; padding: 5rem 1rem; text-align: center; margin-top: 1.5rem; background: rgba(0, 240, 255, 0.02);">
            <div style="font-family: 'Unbounded'; font-size: 1.3rem; color: #00f0ff; letter-spacing: -0.02em;">Awaiting Input Stream</div>
            <div style="font-family: 'Space Grotesk'; font-size: 0.9rem; color: #94a3b8; margin-top: 8px;">Capture a photo with your webcam above or switch to upload mode to launch neural detection.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
