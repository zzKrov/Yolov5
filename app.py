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
# LÍNEA BASE VISUAL: GRADIENTE ANIMADO, FLUIDOS Y PARTÍCULAS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;0,700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #050608 !important;
        color: #e5e7eb !important;
        font-family: 'Spectral', Georgia, serif !important;
        overflow-x: hidden;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* 1. Fondo con gradiente multicolor continuamente animado */
    .stApp {
        background: linear-gradient(135deg, #050608 0%, #0d0f14 25%, #181206 50%, #08090d 75%, #050608 100%) !important;
        background-size: 300% 300% !important;
        animation: gradientShift 16s ease infinite alternate !important;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 2. Esferas orgánicas fluidas de luz en movimiento continuo */
    .fluid-blob-1 {
        position: fixed;
        top: 15%;
        left: -120px;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(212, 175, 55, 0.12) 0%, rgba(245, 158, 11, 0.04) 50%, transparent 70%);
        border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%;
        filter: blur(80px);
        animation: blobMorph1 22s infinite alternate ease-in-out;
        pointer-events: none;
        z-index: 0;
    }

    .fluid-blob-2 {
        position: fixed;
        bottom: 5%;
        right: -100px;
        width: 550px;
        height: 550px;
        background: radial-gradient(circle, rgba(245, 158, 11, 0.1) 0%, rgba(212, 175, 55, 0.03) 50%, transparent 70%);
        border-radius: 60% 40% 30% 70% / 50% 30% 70% 40%;
        filter: blur(90px);
        animation: blobMorph2 25s infinite alternate ease-in-out;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes blobMorph1 {
        0% { transform: translate(0, 0) rotate(0deg) scale(1); }
        50% { transform: translate(120px, 60px) rotate(180deg) scale(1.15); border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; }
        100% { transform: translate(40px, 140px) rotate(360deg) scale(0.95); }
    }

    @keyframes blobMorph2 {
        0% { transform: translate(0, 0) rotate(0deg) scale(1); }
        50% { transform: translate(-100px, -80px) rotate(-180deg) scale(1.2); border-radius: 40% 60% 70% 30% / 50% 60% 30% 60%; }
        100% { transform: translate(-40px, -120px) rotate(-360deg) scale(0.9); }
    }

    /* 3. Tipografía y composición general */
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
        font-size: 2.1rem;
        letter-spacing: 0.02em;
        color: #ffffff;
        text-shadow: 0 0 25px rgba(212, 175, 55, 0.3);
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

    /* Visor de cámara y previsualización proporcionados */
    [data-testid="stCameraInput"] {
        max-width: 480px;
        margin: 0 auto;
    }
    [data-testid="stCameraInput"] video {
        max-height: 280px !important;
        border-radius: 8px !important;
        object-fit: cover !important;
        border: 1px solid rgba(212, 175, 55, 0.35) !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.8) !important;
    }

    .viewport-box {
        background: #000000;
        border: 1px solid rgba(212, 175, 55, 0.4);
        border-radius: 8px;
        padding: 6px;
        max-width: 500px;
        margin: 0 auto;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8), 0 0 20px rgba(212, 175, 55, 0.15);
        position: relative;
        overflow: hidden;
    }

    .scan-beam {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #d4af37, #f59e0b, transparent);
        box-shadow: 0 0 10px #d4af37;
        animation: scanVertical 3s ease-in-out infinite alternate;
        pointer-events: none;
    }

    @keyframes scanVertical {
        0% { top: 4px; opacity: 0.3; }
        50% { opacity: 0.9; }
        100% { top: calc(100% - 6px); opacity: 0.3; }
    }

    .kpi-card {
        background: rgba(9, 11, 16, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 2px solid #d4af37;
        border-radius: 6px;
        padding: 10px 14px;
        backdrop-filter: blur(12px);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #d4af37;
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
        background: rgba(9, 11, 16, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-left: 3px solid #d4af37;
        border-radius: 4px;
        padding: 8px 12px;
        margin-bottom: 6px;
        transition: transform 0.2s ease, border-left-color 0.2s ease;
    }
    .det-item:hover {
        transform: translateX(4px);
        border-left-color: #f59e0b;
    }
    .det-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .det-name {
        font-family: 'Spectral', serif;
        font-weight: 600;
        font-size: 0.92rem;
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
        background: linear-gradient(90deg, #d4af37, #f59e0b);
        box-shadow: 0 0 8px rgba(212, 175, 55, 0.5);
    }
    </style>

    <div class="fluid-blob-1"></div>
    <div class="fluid-blob-2"></div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# PARTÍCULAS DORADAS Y AURA DEL MOUSE
# ---------------------------------------------------------
components.html(
    """
    <script>
    const pDoc = window.parent.document;
    if (!pDoc.getElementById('yolo-particles-canvas')) {
        const canvas = pDoc.createElement('canvas');
        canvas.id = 'yolo-particles-canvas';
        canvas.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:1;opacity:0.6;';
        pDoc.body.prepend(canvas);

        const ctx = canvas.getContext('2d');
        let w = canvas.width = window.parent.innerWidth;
        let h = canvas.height = window.parent.innerHeight;

        window.parent.addEventListener('resize', () => {
            w = canvas.width = window.parent.innerWidth;
            h = canvas.height = window.parent.innerHeight;
        });

        const particles = [];
        for (let i = 0; i < 40; i++) {
            particles.push({
                x: Math.random() * w,
                y: Math.random() * h,
                r: Math.random() * 2 + 0.8,
                vx: (Math.random() - 0.5) * 0.4,
                vy: (Math.random() - 0.5) * 0.4 - 0.2,
                alpha: Math.random() * 0.6 + 0.2
            });
        }

        let mouse = { x: -1000, y: -1000 };
        window.parent.addEventListener('mousemove', (e) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
        });

        function render() {
            ctx.clearRect(0, 0, w, h);
            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];
                p.x += p.vx;
                p.y += p.vy;
                if (p.x < 0) p.x = w;
                if (p.x > w) p.x = 0;
                if (p.y < 0) p.y = h;
                if (p.y > h) p.y = 0;

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(212, 175, 55, ${p.alpha})`;
                ctx.shadowBlur = 6;
                ctx.shadowColor = '#d4af37';
                ctx.fill();

                const d = Math.hypot(p.x - mouse.x, p.y - mouse.y);
                if (d < 100) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(mouse.x, mouse.y);
                    ctx.strokeStyle = `rgba(212, 175, 55, ${0.4 * (1 - d / 100)})`;
                    ctx.lineWidth = 0.8;
                    ctx.stroke();
                }
            }
            requestAnimationFrame(render);
        }
        render();
    }
    </script>
    """,
    height=0,
    width=0
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

h_col1, h_col2, h_col3 = st.columns([2.6, 1.2, 1])
with h_col1:
    st.markdown('<div class="main-title">Detector de Objetos</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-badge">Modelo YOLOv5 // Inferencia óptica calibrada</div>', unsafe_allow_html=True)
with h_col2:
    input_source = st.radio(
        "Fuente de imagen:",
        ["Cámara en vivo", "Subir imagen"],
        horizontal=True,
        label_visibility="collapsed"
    )
with h_col3:
    sound_enabled = st.toggle("Aviso acústico", value=False, help="Emite un tono suave cuando se identifican elementos.")

if model is None:
    st.error("No se pudo cargar el archivo del modelo YOLOv5. Verifica que ultralytics esté instalado.")
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
            "Buscar clases específicas:",
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
            <div style="border: 1px dashed rgba(212, 175, 55, 0.25); border-radius: 6px; padding: 2.5rem 1rem; text-align: center; background: rgba(9, 11, 16, 0.7); backdrop-filter: blur(10px);">
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
            "Modo de vista:",
            options=["Con recuadros", "Imagen original"],
            default="Con recuadros",
            label_visibility="collapsed"
        )

        st.markdown('<div class="viewport-box"><div class="scan-beam"></div>', unsafe_allow_html=True)
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
            st.info("No se identificaron elementos con el nivel de certeza seleccionado.")
    else:
        st.markdown(
            """
            <div style="border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 6px; padding: 2rem 1rem; text-align: center; background: rgba(9, 11, 16, 0.7); backdrop-filter: blur(10px);">
                <div style="font-family: 'JetBrains Mono'; font-size: 0.8rem; color: #9ca3af;">TELEMETRÍA EN REPOSO</div>
                <div style="font-size: 0.78rem; color: #6b7280; margin-top: 4px;">Los resultados de detección y métricas aparecerán aquí tras la captura.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
