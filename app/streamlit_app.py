# app/streamlit_app.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
from io import BytesIO
import base64
from src.pipeline.predict_pipeline import PredictPipeline

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="SoilNet – Soil Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# Custom CSS (dark theme, clean cards, fixed image container)
# ----------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        text-align: center;
    }

    .sub-header {
        font-size: 1.1rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
    }

    .prediction-card {
        background: linear-gradient(135deg, #1e3c2f 0%, #1a2f28 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(74, 222, 128, 0.15);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }

    .prediction-class {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4ade80;
        margin-bottom: 0.5rem;
    }

    .prediction-confidence {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4ade80, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* File uploader area */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(74, 222, 128, 0.3);
        border-radius: 16px;
        padding: 2rem;
        background: rgba(74, 222, 128, 0.03);
    }

    /* Fixed image container – prevents vertical reflow */
    .image-container {
        height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(0,0,0,0.1);
        border-radius: 12px;
        overflow: hidden;
        margin-top: 1rem;
    }
    .image-container img {
        max-height: 100%;
        max-width: 100%;
        object-fit: contain;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Load the model (cached once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    try:
        return PredictPipeline()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

pipeline = load_pipeline()

# ----------------------------------------------------------------------
# Session state – persist results across reruns
# ----------------------------------------------------------------------
if "pred_result" not in st.session_state:
    st.session_state.pred_result = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown('<div class="main-header">🌱 SoilNet</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Classify soil type from an image — Alluvial, Black, Laterite, Red, Yellow, Arid, Mountain</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Two‑column layout
# ----------------------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload a soil image")
    uploaded = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        key="soil_uploader"
    )

    # Detect new upload – reset previous prediction
    if uploaded is not None:
        if st.session_state.uploaded_filename != uploaded.name:
            st.session_state.uploaded_filename = uploaded.name
            st.session_state.pred_result = None
            st.rerun()

    # Display uploaded image in a fixed‑height container
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        # Resize to a max width (aspect ratio preserved)
        basewidth = 400
        wpercent = basewidth / float(image.size[0])
        hsize = int(float(image.size[1]) * float(wpercent))
        image_resized = image.resize((basewidth, hsize), Image.Resampling.LANCZOS)

        # Convert to base64 for stable HTML embedding
        buf = BytesIO()
        image_resized.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        st.markdown(
            f'<div class="image-container">'
            f'<img src="data:image/png;base64,{img_b64}" alt="uploaded soil"/>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        # Placeholder when no image
        st.markdown("""
        <div class="image-container" style="border: 2px dashed rgba(74,222,128,0.15); background: rgba(74,222,128,0.02);">
            <div style="text-align:center; color:#a0aec0;">
                <p style="font-size:3rem; margin-bottom:0;">📸</p>
                <p>Upload an image to classify</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    # Pre‑allocated result slot prevents layout shift
    result_slot = st.empty()

    if uploaded is not None and st.session_state.pred_result is None:
        # Compute prediction only once per image
        image = Image.open(uploaded).convert("RGB")
        img_arr = np.array(image)
        with st.spinner("🔬 Analyzing soil texture..."):
            st.session_state.pred_result = pipeline.predict(img_arr)

    result = st.session_state.pred_result

    # Render into the pre‑allocated slot
    with result_slot.container():
        if result is not None:
            # Prediction Card
            st.markdown(f"""
            <div class="prediction-card">
                <p class="metric-label">PREDICTED SOIL TYPE</p>
                <p class="prediction-class">{result['class']}</p>
                <p class="metric-label">CONFIDENCE</p>
                <p class="prediction-confidence">{result['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

            # Probability Bar Chart
            probs = result["all_probs"]
            df_probs = pd.DataFrame({
                "Soil Type": list(probs.keys()),
                "Confidence": list(probs.values())
            }).sort_values("Confidence", ascending=True)

            fig = px.bar(
                df_probs,
                x="Confidence",
                y="Soil Type",
                orientation="h",
                color="Confidence",
                color_continuous_scale="greens",
                text=df_probs["Confidence"].apply(lambda x: f"{x:.1%}"),
                title="Class Probabilities"
            )
            fig.update_traces(
                textposition="outside",
                marker_line_width=0,
                hovertemplate="%{y}: %{x:.1%}<extra></extra>"
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="",
                showlegend=False,
                height=350,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a0aec0"),
                xaxis=dict(showgrid=False, range=[0, 1]),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Right‑column placeholder when no image
            st.markdown("""
            <div style="height:100%; display:flex; align-items:center; justify-content:center;
                        border:2px dashed rgba(74,222,128,0.15); border-radius:20px;
                        min-height:400px; background: rgba(74,222,128,0.02);">
                <div style="text-align:center; color:#a0aec0;">
                    <p style="font-size:3rem; margin-bottom:0;">📊</p>
                    <p style="font-size:1.1rem;">Prediction will appear here</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.divider()
st.markdown(
    "<div style='text-align:center; color:#a0aec0; font-size:0.85rem;'>"
    "Built with TensorFlow + EfficientNetV2B0 · "
    "<a href='https://github.com/pavanramkalyanboorla-debug/SoilNet' style='color:#4ade80;'>GitHub</a> · "
    "Deployed on Hugging Face Spaces"
    "</div>",
    unsafe_allow_html=True,
)