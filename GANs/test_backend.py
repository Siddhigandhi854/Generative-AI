import io
import base64
import os
import pytest
import numpy as np
import torch
from PIL import Image
from fastapi.testclient import TestClient

from main import app, generator_model
from generator import Generator
from pipeline_utils import postprocess_generated_image

@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client

# 1. TEST GET /health
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data
    assert "device" in data
    assert "weights_file" in data
    assert data["weights_file"] == "generator.pth"

# 2. TEST GET /model-info
def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "DCGAN (Deep Convolutional GAN)"
    assert data["framework"] == "PyTorch 2.x"
    assert data["latent_dimension"] == 100
    assert data["image_dimensions"]["width"] == 64
    assert data["image_dimensions"]["height"] == 64
    assert data["image_dimensions"]["channels"] == 3
    assert data["image_dimensions"]["format"] == "RGB"

# 3. TEST POST /generate
def test_post_generate_endpoint(client):
    response = client.post("/generate?seed=42&response_format=json")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["seed"] == 42
    assert data["latent_dim"] == 100
    assert "image" in data
    assert data["image"].startswith("data:image/png;base64,")
    assert "latent_stats" in data
    assert data["latent_stats"]["seed"] == 42

# 4. TEST POST /generate-batch
def test_post_generate_batch_endpoint(client):
    response = client.post("/generate-batch", json={"count": 4, "seed": 100})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["count"] == 4
    assert data["seed"] == 100
    assert "images" in data
    assert len(data["images"]) == 4
    for b64 in data["images"]:
        assert b64.startswith("data:image/png;base64,")

# 5. TEST INVALID BATCH SIZE (VALIDATION)
def test_invalid_batch_size(client):
    # Test batch size > 16
    resp_over = client.post("/generate-batch", json={"count": 50})
    assert resp_over.status_code == 422
    data_over = resp_over.json()
    assert data_over["error"] == "Validation Error"

    # Test batch size < 1
    resp_under = client.post("/generate-batch", json={"count": 0})
    assert resp_under.status_code == 422
    data_under = resp_under.json()
    assert data_under["error"] == "Validation Error"

# 6. TEST INVALID REQUEST HANDLING
def test_invalid_request_handling(client):
    # Non-existent endpoint
    resp_404 = client.get("/non-existent-endpoint")
    assert resp_404.status_code == 404

    # Malformed JSON payload
    resp_422 = client.post("/generate-batch", content="invalid-json", headers={"Content-Type": "application/json"})
    assert resp_422.status_code == 422

# 7. TEST MODEL LOADING AND INTEGRATION
def test_model_loading_and_weights():
    weights_path = "generator.pth"
    assert os.path.exists(weights_path), f"Checkpoint '{weights_path}' missing"

    device = torch.device("cpu")
    netG = Generator(nz=100, ngf=64, nc=3).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    netG.load_state_dict(state_dict)
    netG.eval()

    # Parameter count verification
    total_params = sum(p.numel() for p in netG.parameters())
    assert total_params == 3576704, f"Expected 3,576,704 params, got {total_params}"

    # Forward pass integration test
    with torch.no_grad():
        z = torch.randn(1, 100, 1, 1, device=device)
        fake = netG(z)

    assert fake.shape == (1, 3, 64, 64)
    assert fake.min().item() >= -1.05 and fake.max().item() <= 1.05

# 8. TEST GENERATED IMAGE VALIDITY
def test_generated_image_validity(client):
    response = client.post("/generate?seed=777&response_format=json")
    assert response.status_code == 200
    b64_uri = response.json()["image"]
    
    b64_data = b64_uri.split(",")[1]
    raw_png_bytes = base64.b64decode(b64_data)

    # Header & structure verification via PIL
    with Image.open(io.BytesIO(raw_png_bytes)) as img:
        img.verify()

# 9. TEST IMAGE DIMENSIONS AND CHANNELS
def test_image_dimensions_and_channels(client):
    response = client.post("/generate?seed=888&response_format=json")
    assert response.status_code == 200
    b64_uri = response.json()["image"]
    
    b64_data = b64_uri.split(",")[1]
    raw_png_bytes = base64.b64decode(b64_data)

    with Image.open(io.BytesIO(raw_png_bytes)) as img:
        assert img.size == (64, 64)
        assert img.mode == "RGB"
        arr = np.array(img)
        assert arr.shape == (64, 64, 3)
        assert arr.dtype == np.uint8
        assert arr.min() >= 0 and arr.max() <= 255
        assert arr.std() > 10.0  # Verify non-blank pixel variance

# 10. TEST API ERROR HANDLING WHEN MODEL IS UNLOADED
def test_api_error_handling_when_model_unloaded(monkeypatch, client):
    import main
    # Simulate model uninitialized/unloaded state
    monkeypatch.setattr(main, "generator_model", None)

    response = client.post("/generate?response_format=json")
    assert response.status_code == 503
    data = response.json()
    assert data["error"] == "API Request Error"
    assert "unavailable" in data["message"].lower() or "not loaded" in data["message"].lower()

    response_batch = client.post("/generate-batch", json={"count": 4})
    assert response_batch.status_code == 503
    data_batch = response_batch.json()
    assert data_batch["error"] == "API Request Error"
