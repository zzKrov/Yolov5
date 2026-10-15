import io
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------
st.set_page_config(
    page_title="Detector de Objetos // Krov Industries",
    page_icon="⚜",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# ESTILOS VISUALES Y CONTROL DE ESCALA
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;0,700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    .stApp {
        background-color: #050608;
        color: #e5e7eb;
        font-family: 'Spectral', Georgia, serif;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .brand-signature {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        color: #9ca3af;
        text-transform: uppercase;
        margin-bottom: 2px;
    }

    .main-title {
        font-family: 'Spectral', serif;
        font-weight: 700;
        font-size: 2rem;
        letter-spacing: 0.02em;
        color: #ffffff;
        margin: 0;
    }

    .sub-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        color: #d4af37;
        text-transform: uppercase;
        margin-top: 2px;
        margin-bottom: 1rem;
    }

    /* Redimensionamiento proporcional de cámara e imágenes */
    [data-testid="stCameraInput"] {
        max-width: 480px;
        margin: 0 auto;
    }
    [data-testid="stCameraInput"] video {
        max-height: 280px !important;
        border-radius: 6px !important;
        object-fit: cover !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
    }

    .viewport-box {
        background: #000000;
        border: 1px solid rgba(212, 175, 55, 0.35);
        border-radius: 6px;
        padding: 6px;
        max-width: 520px;
        margin: 0 auto;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.7), 0 0 15px rgba(212, 175, 55, 0.15);
        position: relative;
    }

    .scan-line {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #d4af37, transparent);
        animation: scanMove 3.5s ease-in-out infinite alternate;
        pointer-events: none;
    }

    @keyframes scanMove {
        0% { top: 4px; opacity: 0.2; }
        50% { opacity: 0.8; }
        100% { top: calc(100% - 6px); opacity: 0.2; }
    }

    .kpi-card {
        background: #090b10;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 2px solid #d4af37;
        border-radius: 4px;
        padding: 10px 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.4);
    }
    .kpi-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #9ca3af;
    }
    .kpi-num {
        font-family: 'Spectral', serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 2px;
    }

    .det-item {
        background: #090b10;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-left: 3px solid #d4af37;
        border-radius: 4px;
        padding: 8px 12px;
        margin-bottom: 6px;
    }
    .det-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 3px;
    }
    .det-name {
        font-family: 'Spectral', serif;
        font-weight: 600;
        font-size: 0.9rem;
        color: #ffffff;
    }
    .det-conf-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #d4af37;
    }
    .det-meter-bg {
        width: 100%;
        height: 4px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 2px;
        overflow: hidden;
    }
    .det-meter-fill {
        height: 100%;
        background: #d4af37;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# CARGA DEL MODELO
# ---------------------------------------------------------
@st.cache_resource
def load_yolo():
    try:
        from ultralytics import YOLO
        return YOLO("yolov5su.pt")
    except Exception:
        return None

model = load_yolo()

# ---------------------------------------------------------
# ENCABEZADO Y CONTROLES SUPERIORES
# ---------------------------------------------------------
st.markdown('<div class="brand-signature">Krov Industries · Visión Artificial</div>', unsafe_allow_html=True)

h_col1, h_col2, h_col3 = st.columns([2.5, 1.2, 1])
with h_col1:
    st.markdown('<div class="main-title">Detector de Objetos</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-badge">Modelo YOLOv5 // Inferencia óptica calibrada</div>', unsafe_allow_html=True)
with h_col2:
    input_source = st.radio(
        "Fuente:",
        ["Cámara en vivo", "Subir imagen"],
        horizontal=True,
        label_visibility="collapsed"
    )
with h_col3:
    sound_enabled = st.toggle("Audio de aviso", value=False, help="Emite un aviso sonoro suave al confirmar detecciones.")

if model is None:
    st.error("No se pudo cargar el modelo YOLOv5. Verifica que la librería ultralytics esté instalada correctamente.")
    st.stop()

# ---------------------------------------------------------
# PARÁMETROS
# ---------------------------------------------------------
with st.expander("Parámetros de detección y filtros de clase", expanded=False):
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1.8])
    with ctrl_col1:
        conf_thresh = st.slider("Certeza mínima", 0.05, 1.0, 0.28, 0.01)
    with ctrl_col2:
        iou_thresh = st.slider("Filtro IoU (Superposición)", 0.10, 1.0, 0.45, 0.05)
    with ctrl_col3:
        all_coco = sorted(list(model.names.values()))
        selected_classes = st.multiselect(
            "Filtrar clases:",
            options=all_coco,
            default=[],
            placeholder="Analizando las 80 clases COCO"
        )

# ---------------------------------------------------------
# CAPTURA DE IMAGEN
# ---------------------------------------------------------
input_image = None
col_camera_side, col_results_side = st.columns([1, 1.1], gap="medium")

with col_camera_side:
    st.markdown("<p style='font-family:JetBrains Mono; font-size:0.75rem; color:#d4af37; margin-bottom:6px;'>ÁREA DE CAPTURA</p>", unsafe_allow_html=True)
    if input_source == "Cámara en vivo":
        shot = st.camera_input("Capturar fotografía", label_visibility="collapsed")
        if shot:
            input_image = Image.open(io.BytesIO(shot.getvalue())).convert("RGB")
    else:
        up = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if up:
            input_image = Image.open(up).convert("RGB")

    if input_image is None:
        st.markdown(
            """
            <div style="border: 1px dashed rgba(212, 175, 55, 0.25); border-radius: 6px; padding: 2.5rem 1rem; text-align: center; background: #07090e;">
                <div style="font-family: 'Spectral'; font-size: 1.1rem; color: #d4af37;">Esperando entrada óptica</div>
                <div style="font-size: 0.8rem; color: #9ca3af; margin-top: 4px;">Captura una fotografía o selecciona un archivo para procesarlo.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# INFERENCIA Y RESULTADOS
# ---------------------------------------------------------
with col_results_side:
    if input_image is not None:
        filter_ids = [k for k, v in model.names.items() if v in selected_classes] if selected_classes else None

        t0 = time.perf_counter()
        results = model(input_image, conf=conf_thresh, iou=iou_thresh, classes=filter_ids)
        inference_ms = (time.perf_counter() - t0) * 1000

        res = results[0]
        boxes = res.boxes
        annotated_rgb = res.plot()[:, :, ::-1]

        count_total = len(boxes) if boxes is not None else 0
        unique_types = len(set(boxes.cls.tolist())) if count_total > 0 else 0
        peak_conf = float(boxes.conf.max().item()) * 100 if count_total > 0 else 0.0

        if sound_enabled and count_total > 0:
            components.html(
                """
                <script>
                (function() {
                    try {
                        const ctx = new (window.AudioContext || window.webkitAudioContext)();
                        const osc = ctx.createOscillator();
                        const gain = ctx.createGain();
                        osc.type = 'sine';
                        osc.frequency.setValueAtTime(587.33, ctx.currentTime);
                        gain.gain.setValueAtTime(0.04, ctx.currentTime);
                        gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.25);
                        osc.connect(gain);
                        gain.connect(ctx.destination);
                        osc.start();
                        osc.stop(ctx.currentTime + 0.25);
                    } catch(e) {}
                })();
                </script>
                """,
                height=0,
                width=0
            )

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Objetos</div><div class="kpi-num" style="color: #d4af37;">{count_total}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Clases</div><div class="kpi-num" style="color: #38bdf8;">{unique_types}</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Tiempo</div><div class="kpi-num" style="color: #9ca3af;">{inference_ms:.0f}<span style="font-size:0.75rem;"> ms</span></div></div>', unsafe_allow_html=True)

        st.write("")
        view_mode = st.segmented_control(
            "Visualización:",
            options=["Con recuadros", "Imagen original"],
            default="Con recuadros",
            label_visibility="collapsed"
        )

        st.markdown('<div class="viewport-box"><div class="scan-line"></div>', unsafe_allow_html=True)
        if view_mode == "Con recuadros":
            st.image(annotated_rgb, use_container_width=True)
        else:
            st.image(input_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        if count_total > 0:
            for b in sorted(boxes, key=lambda x: float(x.conf.item()), reverse=True)[:6]:
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
        else:
            st.info("No se identificaron elementos que superen el umbral configurado.")
    else:
        st.markdown(
            """
            <div style="border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 6px; padding: 2rem 1rem; text-align: center; background: #090b10;">
                <div style="font-family: 'JetBrains Mono'; font-size: 0.8rem; color: #9ca3af;">TELEMETRÍA EN REPOSO</div>
                <div style="font-size: 0.78rem; color: #6b7280; margin-top: 4px;">Los resultados de detección y métricas aparecerán aquí tras la captura.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
