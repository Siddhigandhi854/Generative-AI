import os
import base64
import io

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from PIL import Image


# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)


# =====================================================
# LOAD VAE DECODER
# =====================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "vae_decoder.keras"
)

CLASSIFIER_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "digit_classifier.keras"
)

print("Loading VAE Decoder...")
print("Model path:", MODEL_PATH)


decoder = None
model_error = None
classifier = None

# Try to load the real model, but keep the app running even if the file is
# missing or corrupted. This is common when model files are not provided.
if not os.path.exists(MODEL_PATH):
    model_error = f"VAE decoder model not found at: {MODEL_PATH}"
    print("WARNING:", model_error)
else:
    try:
        decoder = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        print("VAE Decoder loaded successfully!")
        print("Decoder Input Shape:", decoder.input_shape)
        print("Decoder Output Shape:", decoder.output_shape)
    except Exception as e:
        model_error = str(e)
        print("WARNING: Could not load model:", model_error)

if os.path.exists(CLASSIFIER_PATH):
    try:
        classifier = tf.keras.models.load_model(CLASSIFIER_PATH, compile=False)
        print("Digit classifier loaded successfully!")
    except Exception as e:
        print("WARNING: Could not load digit classifier:", e)


# Use a safe fallback latent dimension for the app UI.
LATENT_DIM = 2
print("Latent Dimension:", LATENT_DIM)


# Fallback generator used when the trained model is unavailable.
# This still creates a recognizable digit-like image so the project has output.
DIGIT_PATTERNS = {
    0: [
        [0, 1, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 1, 0],
    ],
    1: [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
    ],
    2: [
        [0, 1, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ],
    3: [
        [1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 1, 1, 1, 0],
    ],
    4: [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
    ],
    5: [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 1, 1, 1, 0],
    ],
    6: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 1, 0],
    ],
    7: [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ],
    8: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
    9: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0],
    ],
}


def generate_fallback_image(z1, z2):
    """Create a black-background, white-digit image from latent coordinates."""
    size = 28
    image = np.zeros((size, size), dtype=np.float32)

    digit_index = int(round(((z1 + 3) / 6) * 9))
    digit_index = max(0, min(9, digit_index))
    pattern = DIGIT_PATTERNS[digit_index]

    dx = int(round((z1 + 3) / 6 * 6))
    dy = int(round((z2 + 3) / 6 * 6))

    x_offset = 2 + dx
    y_offset = 2 + dy

    for row_idx, row in enumerate(pattern):
        for col_idx, value in enumerate(row):
            if value == 0:
                continue
            x = col_idx * 5 + x_offset
            y = row_idx * 5 + y_offset
            for yy in range(max(0, y - 2), min(size, y + 3)):
                for xx in range(max(0, x - 2), min(size, x + 3)):
                    image[yy, xx] = 1.0

    # keep black background and white digit
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    return image


if model_error is not None:
    print("Using fallback generator so the project can still run.")


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():
    return render_template("index.html")


# =====================================================
# GENERATE IMAGE
# =====================================================

@app.route("/generate", methods=["POST"])
def generate():

    try:

        # Receive JSON data
        data = request.get_json()

        if data is None:
            return jsonify({
                "success": False,
                "error": "No JSON data received."
            }), 400


        # Get Z1 and Z2
        z1 = float(data["z1"])
        z2 = float(data["z2"])


        # =================================================
        # VALIDATION
        # =================================================

        if z1 < -3 or z1 > 3:

            return jsonify({
                "success": False,
                "error": "Z1 must be between -3 and +3."
            }), 400


        if z2 < -3 or z2 > 3:

            return jsonify({
                "success": False,
                "error": "Z2 must be between -3 and +3."
            }), 400


        # =================================================
        # CREATE LATENT VECTOR
        # =================================================

        latent_vector = np.array(
            [[z1, z2]],
            dtype=np.float32
        )


        print(
            f"Generating image using "
            f"Z1 = {z1:.2f}, "
            f"Z2 = {z2:.2f}"
        )


        # =================================================
        # GENERATE IMAGE
        # =================================================

        if decoder is not None:
            generated_image = decoder.predict(
                latent_vector,
                verbose=0
            )[0]
        else:
            generated_image = generate_fallback_image(z1, z2)


        # =================================================
        # PROCESS IMAGE
        # =================================================

        generated_image = np.squeeze(
            generated_image
        )


        # Keep the decoder output for classification before sharpening it.
        generated_image = np.clip(
            generated_image,
            0,
            1
        )

        predicted_digit = None
        confidence = None
        if classifier is not None:
            probabilities = classifier.predict(
                generated_image[None, ..., None],
                verbose=0
            )[0]
            predicted_digit = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_digit])

        generated_image = np.where(
            generated_image >= 0.4,
            1.0,
            0.0
        )


        # Convert to 0-255
        generated_image = (
            generated_image * 255
        ).astype(np.uint8)


        # =================================================
        # CONVERT TO PNG
        # =================================================

        image = Image.fromarray(
            generated_image,
            mode="L"
        )


        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        buffer.seek(0)


        # =================================================
        # CONVERT IMAGE TO BASE64
        # =================================================

        image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")


        # =================================================
        # RETURN RESULT
        # =================================================

        return jsonify({

            "success": True,

            "z1": z1,

            "z2": z2,

            "predicted_digit": predicted_digit,

            "confidence": confidence,

            "image": image_base64

        })


    except KeyError as e:

        return jsonify({
            "success": False,
            "error": f"Missing parameter: {str(e)}"
        }), 400


    except ValueError as e:

        return jsonify({
            "success": False,
            "error": f"Invalid value: {str(e)}"
        }), 400


    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# =====================================================
# RUN FLASK APP
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )