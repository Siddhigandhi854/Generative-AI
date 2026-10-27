import io
import os
import sys
import unittest
from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from main import app

class TestGenerativeAILab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_and_attribution(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["guide_teacher"], "Dr. Sachin Takmare")
        self.assertEqual(data["active_experiments"], 10)

    def test_02_diagnostics(self):
        resp = self.client.get("/api/diagnostics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("NVIDIA", data["gpu_accelerator"])

    def test_03_exp1_eda_and_cleaning(self):
        csv_content = b"prompt,score\nHello,0.9\nWorld,\nHello,0.9\nTest,0.5"
        resp = self.client.post(
            "/api/exp1/upload",
            files={"file": ("test.csv", csv_content, "text/csv")}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["rows"], 4)
        self.assertEqual(data["duplicate_rows"], 1)
        self.assertEqual(data["total_missing"], 1)

        # Clean dataset
        clean_resp = self.client.post(
            "/api/exp1/clean",
            data={
                "dataset_id": data["dataset_id"],
                "missing_strategy": "drop",
                "remove_duplicates": "true"
            }
        )
        self.assertEqual(clean_resp.status_code, 200)
        cleaned_data = clean_resp.json()
        self.assertEqual(cleaned_data["rows"], 2)  # 4 - 1 dupe - 1 missing = 2
        self.assertIn("download_url", cleaned_data)

        # Scale features
        scale_resp = self.client.post(
            "/api/exp1/scale",
            data={
                "dataset_id": cleaned_data["dataset_id"],
                "method": "minmax",
                "column": "score"
            }
        )
        self.assertEqual(scale_resp.status_code, 200)

        # Categorical encoding
        encode_resp = self.client.post(
            "/api/exp1/encode",
            data={
                "dataset_id": cleaned_data["dataset_id"],
                "column": "prompt",
                "method": "label"
            }
        )
        self.assertEqual(encode_resp.status_code, 200)

    def test_04_exp2_tensors(self):
        # Create constant tensor
        c_resp = self.client.post("/api/exp2/create", json={
            "creation_type": "constant",
            "shape": [2, 2],
            "values_raw": "[[1, 2], [3, 4]]",
            "dtype": "float32"
        })
        self.assertEqual(c_resp.status_code, 200)
        t_data = c_resp.json()
        self.assertEqual(t_data["shape"], [2, 2])

        # Multiply with another matrix
        op_resp = self.client.post("/api/exp2/operate", json={
            "op_type": "matmul",
            "tensor_a": [[1, 2], [3, 4]],
            "tensor_b": [[2, 0], [1, 2]]
        })
        self.assertEqual(op_resp.status_code, 200)
        op_data = op_resp.json()
        self.assertEqual(op_data["result"], [[4.0, 4.0], [10.0, 8.0]])

        # Test activations: ReLU
        relu_resp = self.client.post("/api/exp2/operate", json={
            "op_type": "relu",
            "tensor_a": [[-2.0, 3.0], [0.0, -5.0]]
        })
        self.assertEqual(relu_resp.status_code, 200)
        self.assertEqual(relu_resp.json()["result"], [[0.0, 3.0], [0.0, 0.0]])

        # Test reductions: Reduce Mean
        mean_resp = self.client.post("/api/exp2/operate", json={
            "op_type": "reduce_mean",
            "tensor_a": [[2.0, 4.0], [6.0, 8.0]]
        })
        self.assertEqual(mean_resp.status_code, 200)
        self.assertEqual(mean_resp.json()["result"], 5.0)

        # Test Linear Algebra: Determinant
        det_resp = self.client.post("/api/exp2/operate", json={
            "op_type": "det",
            "tensor_a": [[4.0, 2.0], [3.0, 3.0]]
        })
        self.assertEqual(det_resp.status_code, 200)
        self.assertAlmostEqual(det_resp.json()["result"], 6.0, places=1)

        # Test Shape Manipulation: Reshape
        reshape_resp = self.client.post("/api/exp2/operate", json={
            "op_type": "reshape",
            "tensor_a": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "target_shape": [3, 2]
        })
        self.assertEqual(reshape_resp.status_code, 200)
        self.assertEqual(reshape_resp.json()["shape"], [3, 2])

    def test_04b_exp3_transformer_qa(self):
        # Status endpoint
        status_resp = self.client.get("/api/exp3/status")
        self.assertEqual(status_resp.status_code, 200)
        s_data = status_resp.json()
        self.assertIn("default_model", s_data)

        # Generate QA answer with context
        gen_resp = self.client.post("/api/exp3/generate", json={
            "question": "What does the Transformer architecture rely entirely on?",
            "context": "The Transformer relies entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned RNNs.",
            "provider": "huggingface",
            "model_name": "deepset/roberta-base-squad2"
        })
        self.assertEqual(gen_resp.status_code, 200)
        g_data = gen_resp.json()
        self.assertIn("answer", g_data)
        self.assertIn("self-attention", g_data["answer"].lower())
        self.assertIsNotNone(g_data["start_char"])
        self.assertIsNotNone(g_data["end_char"])

    def test_04c_exp4_diffusion(self):
        # Generate image
        gen_resp = self.client.post("/api/exp4/generate", json={
            "prompt": "Futuristic cyberpunk city street at dusk with neon signage reflecting on rain-slicked asphalt",
            "aspect_ratio": "1:1",
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0"
        })
        self.assertEqual(gen_resp.status_code, 200)
        data = gen_resp.json()
        self.assertIn("image_url", data)
        self.assertIn("dimensions", data)

        # Download image
        img_resp = self.client.get(data["image_url"])
        self.assertEqual(img_resp.status_code, 200)
        self.assertEqual(img_resp.headers["content-type"], "image/png")

    def test_05_exp5_vae(self):
        resp = self.client.post("/api/exp5/generate", json={
            "num_samples": 4,
            "seed": 42
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["samples"]), 4)
        img_url = data["samples"][0]["image_url"]

        # Verify image download
        img_resp = self.client.get(img_url)
        self.assertEqual(img_resp.status_code, 200)
        self.assertEqual(img_resp.headers["content-type"], "image/png")

    def test_06_exp6_dcgan(self):
        resp = self.client.post("/api/exp6/generate", json={
            "num_samples": 2,
            "seed": 100
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["samples"]), 2)
        img_url = data["samples"][0]["image_url"]

        img_resp = self.client.get(img_url)
        self.assertEqual(img_resp.status_code, 200)
        self.assertEqual(img_resp.headers["content-type"], "image/png")

    def test_07_exp7_stylegan(self):
        resp = self.client.post("/api/exp7/generate", json={
            "num_samples": 2,
            "seed": 200,
            "truncation_psi": 0.6
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["samples"]), 2)
        img_url = data["samples"][0]["image_url"]

        img_resp = self.client.get(img_url)
        self.assertEqual(img_resp.status_code, 200)
        self.assertEqual(img_resp.headers["content-type"], "image/png")

    def test_08_exp8_musicgen(self):
        resp = self.client.post("/api/exp8/generate", json={
            "prompt": "Peaceful instrumental music with piano and soft strings.",
            "duration_seconds": 3
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("audio_url", data)

        audio_resp = self.client.get(data["audio_url"])
        self.assertEqual(audio_resp.status_code, 200)
        self.assertEqual(audio_resp.headers["content-type"], "audio/wav")

    def test_09_exp9_rag_and_finetune(self):
        # Upload doc
        doc_content = b"The guide teacher is Dr. Sachin Takmare. The lab covers 10 GenAI experiments."
        up_resp = self.client.post(
            "/api/exp9/upload",
            files={"file": ("guide_info.txt", doc_content, "text/plain")}
        )
        self.assertEqual(up_resp.status_code, 200)

        # Query chat
        chat_resp = self.client.post("/api/exp9/chat", json={
            "question": "Who is the guide teacher?",
            "top_k": 2
        })
        self.assertEqual(chat_resp.status_code, 200)
        cdata = chat_resp.json()
        self.assertIn("Sachin Takmare", cdata["answer"])
        self.assertGreater(len(cdata["sources"]), 0)

        # LoRA simulation
        ft_resp = self.client.post("/api/exp9/finetune-simulate", json={
            "model_name": "meta-llama/Llama-3-8B",
            "lora_r": 8,
            "lora_alpha": 16,
            "epochs": 2
        })
        self.assertEqual(ft_resp.status_code, 200)
        ft_data = ft_resp.json()
        self.assertIn("trainable_params", ft_data)

    def test_10_exp10_video_generation(self):
        resp = self.client.post("/api/exp10/generate", json={
            "prompt": "A cinematic drone shot flying over green mountains at sunrise.",
            "num_frames": 12,
            "fps": 8
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("video_url", data)

        video_resp = self.client.get(data["video_url"])
        self.assertEqual(video_resp.status_code, 200)
        self.assertEqual(video_resp.headers["content-type"], "video/mp4")

if __name__ == "__main__":
    unittest.main()
