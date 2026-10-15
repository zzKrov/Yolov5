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
# FOCO AMBIENTAL SUTIL
# ---------------------------------------------------------
components.html(
    """
    <script>
    const pDoc = window.parent.document;
    let aura = pDoc.getElementById('krov-mouse-aura');
    if (!aura) {
        aura = pDoc.createElement('div');
        aura.id = 'krov-mouse-aura';
        aura.style.cssText = 'position:fixed;pointer-events:none;width:380px;height:380px;border-radius:50%;background:radial-gradient(circle, rgba(212, 175, 55, 0.07) 0%, transparent 65%);transform:translate(-50%, -50%);z-index:999999;mix-blend-mode:screen;filter:blur(14px);transition:transform 0.04s ease-out;';
        pDoc.body.appendChild(aura);
    }
    pDoc.addEventListener('mousemove', (e) => {
        aura.style.left = e.clientX + 'px';
        aura.style.top = e.clientY + 'px';
    });
    </script>
    """,
    height=0,
    width=0
)

# ---------------------------------------------------------
# ESTILOS VISUALES
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,300;0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap');

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
        margin-bottom: 4px;
    }

    .main-title {
        font-family: 'Spectral', serif;
        font-weight: 700;
        font-size: 2.2rem;
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
        margin-top: 4px;
        margin-bottom: 1.2rem;
    }

    .kpi-card {
        background: #090b10;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 2px solid #d4af37;
        border-radius: 4px;
        padding: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.4);
    }
    .kpi-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #9ca3af;
    }
    .kpi-num {
        font-family: 'Spectral', serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 2px;
    }

    .viewport-frame {
        background: #000000;
        border: 1px solid rgba(212, 175, 55, 0.25);
        border-radius: 4px;
        padding: 6px;
    }

    .det-item {
        background: #090b10;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-left: 3px solid #d4af37;
        border-radius: 4px;
        padding: 10px 12px;
        margin-bottom: 8px;
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
        font-size: 0.95rem;
        color: #ffffff;
    }
    .det-conf-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
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
# ENCABEZADO
# ---------------------------------------------------------
st.markdown('<div class="brand-signature">Krov Industries · Visión Artificial</div>', unsafe_allow_html=True)

head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.markdown('<div class="main-title">Detector de Objetos</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-badge">Modelo YOLOv5 // Identificación en tiempo real y fotografías</div>',
        unsafe_allow_html=True
    )

with head_col2:
    input_source = st.radio(
        "Origen de imagen:",
        ["Cámara en vivo", "Subir imagen"],
        horizontal=True,
        label_visibility="collapsed"
    )

if model is None:
    st.error("No se pudo cargar el modelo YOLOv5. Asegúrate de tener instalada la librería ultralytics en el entorno.")
    st.stop()

# ---------------------------------------------------------
# PANEL DE CONTROL
# ---------------------------------------------------------
with st.expander("Ajustes de detección y filtros de clase", expanded=True):
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 2])

    with ctrl_col1:
        conf_thresh = st.slider(
            "Certeza mínima",
            0.05, 1.0, 0.28, 0.01,
            help="Los objetos detectados con una confianza inferior a este umbral serán descartados."
        )

    with ctrl_col2:
        iou_thresh = st.slider(
            "Filtro de superposición (IoU)",
            0.10, 1.0, 0.45, 0.05,
            help="Reduce recuadros duplicados sobre el mismo elemento detectado."
        )

    with ctrl_col3:
        all_coco = sorted(list(model.names.values()))
        selected_classes = st.multiselect(
            "Filtrar por clases específicas:",
            options=all_coco,
            default=[],
            placeholder="Detectando todas las clases disponibles"
        )

# ---------------------------------------------------------
# CAPTURA DE IMAGEN
# ---------------------------------------------------------
input_image = None

if input_source == "Cámara en vivo":
    shot = st.camera_input("Capturar fotografía con la cámara", label_visibility="collapsed")
    if shot:
        input_image = Image.open(io.BytesIO(shot.getvalue())).convert("RGB")
else:
    up = st.file_uploader(
        "Cargar archivo de imagen",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
    if up:
        input_image = Image.open(up).convert("RGB")

# ---------------------------------------------------------
# INFERENCIA Y RESULTADOS
# ---------------------------------------------------------
if input_image is not None:
    filter_ids = [k for k, v in model.names.items() if v in selected_classes] if selected_classes else None

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

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Objetos encontrados</div><div class="kpi-num" style="color: #d4af37;">{count_total}</div></div>',
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Clases distintas</div><div class="kpi-num" style="color: #38bdf8;">{unique_types}</div></div>',
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Mayor confianza</div><div class="kpi-num" style="color: #f59e0b;">{peak_conf:.1f}%</div></div>',
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Tiempo de análisis</div><div class="kpi-num" style="color: #9ca3af;">{inference_ms:.1f}<span style="font-size:0.8rem; font-family:JetBrains Mono;"> ms</span></div></div>',
            unsafe_allow_html=True
        )

    st.write("")

    col_viewport, col_telemetry = st.columns([1.35, 1], gap="large")

    with col_viewport:
        view_mode = st.segmented_control(
            "Modo de visualización:",
            options=["Con recuadros", "Imagen original"],
            default="Con recuadros",
            label_visibility="collapsed"
        )

        st.markdown('<div class="viewport-frame">', unsafe_allow_html=True)
        if view_mode == "Con recuadros":
            st.image(annotated_rgb, use_container_width=True)
        else:
            st.image(input_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_telemetry:
        st.markdown("<h4 style='font-family:Spectral; margin-top:0; color:#d4af37;'>Distribución de detecciones</h4>", unsafe_allow_html=True)

        if count_total > 0:
            cat_counts = {}
            for b in boxes:
                c = int(b.cls.item())
                cat_counts[c] = cat_counts.get(c, 0) + 1

            df_chart = pd.DataFrame({
                "Clase": [model.names[c].capitalize() for c in cat_counts.keys()],
                "Total": list(cat_counts.values())
            }).sort_values(by="Total", ascending=False)

            st.bar_chart(df_chart.set_index("Clase"), color="#d4af37")
            st.write("")

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
                st.caption(f"Mostrando los primeros 8 de {count_total} elementos identificados.")
        else:
            st.markdown(
                """
                <div style="border: 1px solid rgba(212, 175, 55, 0.3); border-radius: 4px; padding: 1.5rem; text-align: center; background: #090b10;">
                    <div style="font-family: Spectral; font-size: 1rem; color: #d4af37; margin-bottom: 4px;">No se identificaron objetos</div>
                    <div style="font-size: 0.85rem; color: #9ca3af;">Prueba reduciendo el umbral de certeza mínima en el panel superior.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.markdown(
        """
        <div style="border: 1px dashed rgba(212, 175, 55, 0.25); border-radius: 6px; padding: 3.5rem 1rem; text-align: center; margin-top: 1rem; background: #07090e;">
            <div style="font-family: 'Spectral'; font-size: 1.2rem; color: #d4af37;">Esperando una imagen para procesar</div>
            <div style="font-size: 0.85rem; color: #9ca3af; margin-top: 4px;">Toma una foto con la cámara o sube un archivo local para iniciar la detección.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
