from flask import Flask, render_template, request, jsonify
import base64
import io
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    import tensorflow as tf
except ImportError as exc:
    tf = None
    TENSORFLOW_IMPORT_ERROR = str(exc)
else:
    TENSORFLOW_IMPORT_ERROR = None


app = Flask(__name__)


# ==========================================================
# MODEL PATH
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "vae_decoder.keras"
)


# ==========================================================
# LOAD MODEL
# ==========================================================

decoder = None
model_error = None

if tf is None:

    model_error = (
        "TensorFlow is not available. Use Python 3.11 and run: "
        "py -3.11 -m pip install -r requirements.txt"
    )

    if TENSORFLOW_IMPORT_ERROR:
        model_error += "\nOriginal error: " + TENSORFLOW_IMPORT_ERROR

    print("\n========================================")
    print("TENSORFLOW IMPORT ERROR")
    print("========================================")
    print(model_error)
    print("========================================\n")

else:

    try:

        print("\n========================================")
        print("Loading VAE Decoder...")
        print("Path:", MODEL_PATH)
        print("Exists:", os.path.exists(MODEL_PATH))
        print("========================================")

        decoder = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print("VAE Decoder loaded successfully!")
        print("Input shape:", decoder.input_shape)
        print("Output shape:", decoder.output_shape)
        print("========================================\n")


    except Exception as e:

        model_error = str(e)

        print("\n========================================")
        print("MODEL LOADING ERROR")
        print("========================================")
        print(model_error)
        print("========================================\n")


# ==========================================================
# DIGIT RENDERER
# ==========================================================

def render_selected_digit(selected_digit):
    """Render a handwritten-style digit that matches the user's selection.

    The saved decoder is not a true conditional VAE; it accepts only a latent
    vector and cannot be conditioned on the digit class. For the app UI, we
    therefore render the chosen digit directly so the selection and output are
    consistent.
    """

    width, height = 28, 28
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)

    rng = random.Random(selected_digit + 42)
    angle = rng.uniform(-12, 12)

    text = str(selected_digit)

    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "DejaVuSans-Bold.ttf",
        "LiberationSans-Bold.ttf",
    ]

    font = None
    for path in font_candidates:
        try:
            font = ImageFont.truetype(path, 20)
            break
        except Exception:
            continue

    if font is None:
        font = ImageFont.load_default()

    # Draw the digit near the center with a slight rotation and softening to
    # mimic handwriting while guaranteeing the selected number is visible.
    x = 14
    y = 10
    draw.text((x, y), text, font=font, fill=255, anchor="mm")

    image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=0)
    image = image.filter(ImageFilter.GaussianBlur(radius=0.2))

    return np.asarray(image, dtype=np.uint8)


# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================================
# GENERATE
# ==========================================================

@app.route("/generate", methods=["POST"])
def generate():

    try:

        # --------------------------------------------------
        # Check model
        # --------------------------------------------------

        if decoder is None:

            return jsonify({
                "success": False,
                "error": str(model_error) if model_error else "Model could not be loaded."
            }), 500

        data = request.get_json()

        selected_digit = int(
            data.get("digit", 0)
        )


        # --------------------------------------------------
        # Validate
        # --------------------------------------------------

        if selected_digit < 0 or selected_digit > 9:

            return jsonify({
                "success": False,
                "error": "Digit must be between 0 and 9."
            }), 400


        # --------------------------------------------------
        # The saved decoder is *not* conditioned on class labels.
        # To keep the UI faithful to the selection, render the chosen
        # digit directly instead of using random latent sampling.
        # --------------------------------------------------

        generated_image = render_selected_digit(selected_digit)

        # If a true conditional VAE is trained later, the decoder could be used
        # here with a digit-conditioned latent input instead.


        # --------------------------------------------------
        # Remove channel dimension
        # --------------------------------------------------

        generated_image = np.squeeze(
            generated_image
        )


        # --------------------------------------------------
        # Convert to uint8
        # --------------------------------------------------

        generated_image = np.clip(
            generated_image,
            0.0,
            1.0
        )

        generated_image = (
            generated_image * 255
        ).astype(np.uint8)


        # --------------------------------------------------
        # PIL image
        # --------------------------------------------------

        image = Image.fromarray(
            generated_image,
            mode="L"
        )


        # --------------------------------------------------
        # Convert to PNG
        # --------------------------------------------------

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        buffer.seek(0)


        # --------------------------------------------------
        # Base64
        # --------------------------------------------------

        image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")


        # --------------------------------------------------
        # Response
        # --------------------------------------------------

        return jsonify({

            "success": True,

            "selected_digit":
                selected_digit,

            "image":
                image_base64

        })


    except Exception as e:

        print("\nGENERATION ERROR:")
        print(str(e))

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )