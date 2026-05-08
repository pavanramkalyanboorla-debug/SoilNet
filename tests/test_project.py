import requests
import numpy as np
from PIL import Image
import io
import os

BASE = "http://localhost:8000"


def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    data = r.json()
    assert data["model_loaded"] is True, "Model not loaded"
    assert data["num_classes"] == 7, f"Expected 7 classes, got {data['num_classes']}"
    print("[PASS] Health check")


def test_predict():
    # Create a dummy image (random pixels)
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    r = requests.post(f"{BASE}/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert r.status_code == 200, f"Prediction failed: {r.text}"
    data = r.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1
    assert len(data["all_probabilities"]) == 7
    print(f"[PASS] Predict: {data['predicted_class']} ({data['confidence']:.2%})")


def test_predict_batch():
    """Test that repeated predictions are consistent."""
    np.random.seed(42)
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    results = []
    for _ in range(3):
        buf.seek(0)
        r = requests.post(f"{BASE}/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        results.append(r.json()["predicted_class"])

    # All three predictions should be the same
    assert len(set(results)) == 1, f"Inconsistent predictions: {results}"
    print(f"[PASS] Batch consistency: all 3 predictions = {results[0]}")


if __name__ == "__main__":
    test_health()
    test_predict()
    test_predict_batch()
    print("\n✅ All tests passed.")