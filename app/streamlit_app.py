import sys
import os
from io import BytesIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import plotly.express as px

# ----------------------------------------------------------------------
# Streamlit config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="SoilNet – Soil Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# IMPORTANT HF SPACE FIXES
# ----------------------------------------------------------------------
# Prevent oversized uploads crashing HF proxy
MAX_IMAGE_SIZE = (512, 512)

# ----------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------
st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.25rem;
        letter-spacing: -1px;
    }

    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2.5rem;
    }

    .prediction-card {
        background: linear-gradient(
            135deg,
            rgba(30, 41, 59, 0.95),
            rgba(15, 23, 42, 0.98)
        );
        border: 1px solid rgba(74, 222, 128, 0.15);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }

    .prediction-class {
        font-size: 2.5rem;
        font-weight: 800;
        color: #4ade80;
        margin-bottom: 0.5rem;
    }

    .prediction-confidence {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4ade80, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(74, 222, 128, 0.25);
        border-radius: 18px;
        padding: 1.5rem;
        background: rgba(74, 222, 128, 0.03);
    }

    .footer {
        text-align:center;
        color:#94a3b8;
        font-size:0.85rem;
        margin-top:2rem;
    }

    .footer a {
        color:#4ade80;
        text-decoration:none;
    }

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Lazy model loading (better for HF Spaces RAM)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from src.pipeline.predict_pipeline import PredictPipeline
    return PredictPipeline()


# ----------------------------------------------------------------------
# Safe image preprocessing
# ----------------------------------------------------------------------
def preprocess_uploaded_image(uploaded_file):
    """
    Robust in-memory image preprocessing for HF Spaces.
    Prevents upload crashes and EXIF orientation bugs.
    """

    try:
        image_bytes = uploaded_file.read()

        image = Image.open(BytesIO(image_bytes))

        # Fix rotated phone images
        image = ImageOps.exif_transpose(image)

        # Force RGB
        image = image.convert("RGB")

        # Aggressive resize for HF stability
        image.thumbnail(MAX_IMAGE_SIZE)

        return image

    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        return None


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    '<div class="main-header">🌱 SoilNet</div>',
    unsafe_allow_html=True
)

st.markdown(
    '''
    <div class="sub-header">
        Deep learning soil classification using EfficientNetV2B0
    </div>
    ''',
    unsafe_allow_html=True
)

# ----------------------------------------------------------------------
# Load model safely
# ----------------------------------------------------------------------
try:
    pipeline = load_pipeline()

except Exception as e:
    st.error("Failed to load model artifacts.")
    st.exception(e)
    st.stop()

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

# ----------------------------------------------------------------------
# LEFT PANEL
# ----------------------------------------------------------------------
with col1:

    st.markdown("### 📤 Upload Soil Image")

    uploaded = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="soil_uploader",
    )

    st.caption(
        "Supported formats: JPG, JPEG, PNG • Max recommended size: 5MB"
    )

    if uploaded is not None:

        image = preprocess_uploaded_image(uploaded)

        if image is not None:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

# ----------------------------------------------------------------------
# RIGHT PANEL
# ----------------------------------------------------------------------
with col2:

    if uploaded is not None:

        image = preprocess_uploaded_image(uploaded)

        if image is not None:

            img_arr = np.array(image)

            try:

                with st.spinner("🔬 Analyzing soil texture..."):

                    result = pipeline.predict(img_arr)

                # ------------------------------------------------------
                # Prediction card
                # ------------------------------------------------------
                st.markdown(f"""
                <div class="prediction-card">

                    <p class="metric-label">
                        Predicted Soil Type
                    </p>

                    <p class="prediction-class">
                        {result['class']}
                    </p>

                    <p class="metric-label">
                        Confidence
                    </p>

                    <p class="prediction-confidence">
                        {result['confidence']:.1%}
                    </p>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ------------------------------------------------------
                # Probability chart
                # ------------------------------------------------------
                probs = result["all_probs"]

                df_probs = pd.DataFrame({
                    "Soil Type": list(probs.keys()),
                    "Confidence": list(probs.values())
                })

                df_probs = df_probs.sort_values(
                    "Confidence",
                    ascending=True
                )

                fig = px.bar(
                    df_probs,
                    x="Confidence",
                    y="Soil Type",
                    orientation="h",
                    text=df_probs["Confidence"].apply(
                        lambda x: f"{x:.1%}"
                    ),
                    color="Confidence",
                    color_continuous_scale="greens",
                    title="Class Probabilities",
                )

                fig.update_traces(
                    textposition="outside",
                    marker_line_width=0,
                    hovertemplate="%{y}: %{x:.1%}<extra></extra>"
                )

                fig.update_layout(
                    height=360,
                    margin=dict(l=0, r=0, t=50, b=0),
                    xaxis_title="",
                    yaxis_title="",
                    showlegend=False,
                    xaxis=dict(
                        range=[0, 1],
                        showgrid=False,
                    ),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1"),
                    coloraxis_showscale=False,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as e:

                st.error("Prediction failed.")
                st.exception(e)

    else:

        st.markdown("""
        <div style="
            min-height:420px;
            display:flex;
            align-items:center;
            justify-content:center;
            border:2px dashed rgba(74,222,128,0.15);
            border-radius:24px;
            background:rgba(74,222,128,0.02);
        ">
            <div style="text-align:center;color:#94a3b8;">

                <p style="
                    font-size:4rem;
                    margin-bottom:0.5rem;
                ">
                    📸
                </p>

                <p style="
                    font-size:1.1rem;
                ">
                    Upload an image to begin prediction
                </p>

            </div>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.divider()

st.markdown("""
<div class="footer">
    Built with TensorFlow + EfficientNetV2B0 ·
    <a href="https://github.com/pavanramkalyanboorla-debug/SoilNet" target="_blank">
        GitHub
    </a>
    · Deployed on Hugging Face Spaces
</div>
""", unsafe_allow_html=True)