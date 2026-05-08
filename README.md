---
title: SoilNet
emoji: 🌱
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🌱 SoilNet — Soil Type Classification with Deep Learning

**End‑to‑end lightweight CNN for soil classification — trained on CPU, deployed on Hugging Face Spaces.**

[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-HuggingFace-yellow)](https://huggingface.co/spaces/PavanBoorla/soil-classifier)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What It Does

SoilNet classifies soil types from a single image into one of **7 categories**:

| Soil Type | | Soil Type |
|-----------|---|-----------|
| 🟤 Alluvial | | ⚫ Black |
| 🔴 Red | | 🟡 Yellow |
| 🟠 Laterite | | 🏜️ Arid |
| ⛰️ Mountain | | |

It uses a pre‑trained **EfficientNetV2B0** backbone fine‑tuned on a public soil dataset. The entire system is designed to **train on a CPU laptop** and **deploy as a lightweight Docker container** — no GPU required.

---

## 📸 Live Demo

Try it yourself: **upload any soil photo** and get an instant prediction with confidence scores.

👉 [**Live Demo on Hugging Face Spaces**](https://huggingface.co/spaces/PavanBoorla/soil-classifier)

---

## 🧠 Why This Project?

Coming from a **civil engineering background**, I've always been interested in how machine learning can solve real-world geotechnical and agricultural problems. Soil classification is usually done by experts in the field — but it's slow, subjective, and doesn't scale.

This project was my introduction to **computer vision** and **transfer learning**. I intentionally kept the hardware constraint realistic: training and inference both run on a normal laptop without a dedicated GPU.

The result is a **production‑ready ML system** that demonstrates:

- ✅ Transfer learning with EfficientNetV2B0
- ✅ Proper train/validation stratification on a small, imbalanced dataset
- ✅ Class‑weighted training to handle imbalance
- ✅ Fine‑tuning of the top 20% of the backbone
- ✅ Comprehensive evaluation (precision, recall, F1, confusion matrix)
- ✅ Inference latency & model size benchmarking
- ✅ Multi‑stage Docker build with `uv`
- ✅ FastAPI + Streamlit serving
- ✅ Polished, recruiter‑friendly Streamlit UI

---

## 📊 Performance

Evaluated on a held‑out validation set of **238 images** (20% of the dataset).

**Overall accuracy:** **84.5%**
**Weighted average F1‑score:** **0.84**

| Class    | Precision | Recall | F1‑score | Support |
|----------|-----------|--------|----------|---------|
| Alluvial | 0.57      | 0.40   | 0.47     | 10      |
| Arid     | 0.81      | 0.95   | 0.87     | 57      |
| Black    | 0.94      | 0.90   | 0.92     | 51      |
| Laterite | 0.93      | 0.57   | 0.70     | 44      |
| Mountain | 0.85      | 0.98   | 0.91     | 40      |
| Red      | 0.73      | 1.00   | 0.85     | 22      |
| Yellow   | 0.92      | 0.79   | 0.85     | 14      |

**Key insights:**
- Strong performance on visually distinct classes (Black, Arid, Mountain).
- Misclassifications occur between visually similar soils (Laterite ↔ Red, Alluvial ↔ Mountain) — expected even for human experts.
- Class weighting significantly improved recall on minority classes (Red, Yellow).

Full evaluation artifacts (`confusion_matrix`, `metrics.json`, `benchmark.json`) are available in the `evaluation/` folder.

---

## 🏗️ Architecture

1. **Data pipeline** — Manual stratified train/validation split, `tf.data.Dataset` with caching, prefetching, and pixel‑level augmentation.
2. **Model** — EfficientNetV2B0 backbone (frozen during initial training, then top 20% unfrozen for fine‑tuning) + custom classification head.
3. **Training** — Two‑phase: head‑only training with class weights, then fine‑tuning with a very low learning rate.
4. **Export** — Trained weights saved in `.weights.h5` (HDF5) format for version‑independent deployment.
5. **Inference** — A separate `PredictPipeline` rebuilds the model architecture from scratch and loads only the weights, avoiding Keras serialization issues.
6. **Serving** — FastAPI backend with a `/predict` endpoint and a polished Streamlit UI.
7. **Deployment** — Multi‑stage Docker image, pushed to Hugging Face Spaces (SDK: Docker).

---

## ⚡ Quick Start (Local)

### Prerequisites

- Python ≥ 3.10
- `uv` installed ([instructions](https://docs.astral.sh/uv/getting-started/installation/))

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/pavanramkalyanboorla-debug/soil-classifier.git
cd soil-classifier

# 2. Download the dataset from Kaggle and place it in data/soil_dataset/
#    (should contain subfolders: Alluvial_Soil, Arid_Soil, Black_Soil, etc.)

# 3. Train the model (optional — pre‑trained weights are already in artifacts/)
uv run python src/pipeline/training_pipeline.py

# 4. Export model weights (if you retrained)
uv run python -c "import tensorflow as tf; model = tf.keras.models.load_model('artifacts/model.keras'); model.save_weights('artifacts/model_weights.weights.h5')"

# 5. Start the API
uvicorn app.main:app --reload --port 8000

# 6. Or run the Streamlit demo
streamlit run app/streamlit_app.py --server.port 7860
```

---

## 🐳 Docker

A pre‑built Docker image is available for local testing:

```bash
docker build -t soilnet .
docker run -p 7860:7860 soilnet
```

The image includes the trained weights and is ready to use. **No GPU required.**

---

## 🚀 Deployment

The live demo on Hugging Face Spaces is updated by pushing to the Space's Git remote:

```bash
git remote add space https://huggingface.co/spaces/PavanBoorla/soil-classifier
git push space main --force
```

> **Important:** The large binary artifacts (`model_weights.weights.h5`, `class_names.pkl`) are not stored in Git. They are uploaded separately via the Hugging Face UI (**File → Add file → Upload**). The Docker build copies them from the Space's persistent storage.

---

## 📂 Project Structure

```
soil-classifier/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI endpoint
│   └── streamlit_app.py     # Streamlit UI (polished)
├── src/
│   ├── components/
│   │   ├── data_loader.py       # Stratified data loading
│   │   ├── preprocessing.py     # Caching & augmentation
│   │   ├── model_trainer.py     # Two‑phase training
│   │   └── model_evaluator.py   # Metrics & benchmarking
│   ├── pipeline/
│   │   ├── training_pipeline.py # Orchestrates training
│   │   └── predict_pipeline.py  # Inference (weight‑loading)
│   ├── utils/
│   │   ├── exceptions.py
│   │   ├── logger.py
│   │   └── utils.py
│   └── constants.py             # All paths & hyperparameters
├── artifacts/                   # Weights & class names (gitignored)
│   ├── model_weights.weights.h5
│   ├── class_names.pkl
│   └── evaluation/
├── notebooks/                   # Jupyter exploration notebook
├── tests/
│   └── test_project.py
├── pyproject.toml
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 🎓 What I Learned

- How to design a deep learning pipeline under CPU constraints.
- The importance of stratified splitting for small, imbalanced datasets.
- How to handle Keras model serialization issues by using plain weights (`.h5`) and rebuilding the architecture at load time.
- Production‑ready Docker patterns — multi‑stage builds, `uv`, and separating artifacts from code.
- How to build an attractive, professional Streamlit UI that demonstrates the entire system to recruiters.

---

## 📄 License

[MIT](https://opensource.org/licenses/MIT) — feel free to use, modify, and deploy.

---

## 👤 Author

**Boorla Pavan Ram Kalyan**

[![GitHub](https://img.shields.io/badge/GitHub-pavanramkalyanboorla--debug-181717?logo=github)](https://github.com/pavanramkalyanboorla-debug)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com)