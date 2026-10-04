from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

import os
import uuid

from model import generate_faces, save_image


app = Flask(__name__)


GENERATED_FOLDER = os.path.join(
    "static",
    "generated"
)

os.makedirs(
    GENERATED_FOLDER,
    exist_ok=True
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(silent=True) or {}

        num_images = int(data.get("num_images", 1))
        num_images = max(1, min(num_images, 8))

        skin_tone = float(data.get("skin_tone", 0.5))
        face_age = float(data.get("face_age", 0.5))
        expression = float(data.get("expression", 0.5))
        variation = float(data.get("variation", 0.5))

        truncation = float(data.get("truncation", 0.7))
        truncation = max(0.5, min(truncation, 1.0))

        seed_value = data.get("seed")
        if seed_value in [None, "", "random"]:
            seed = None
        else:
            seed = int(seed_value)

        if seed is not None:
            seed = (seed + int(skin_tone * 1000) + int(face_age * 100) + int(expression * 1000) + int(variation * 10000)) % (2 ** 31)
        else:
            seed = int((skin_tone * 997 + face_age * 131 + expression * 401 + variation * 733) * 1000)

        images = generate_faces(
            num_images=num_images,
            truncation=truncation,
            seed=seed,
            skin_tone=skin_tone,
            face_age=face_age,
            expression=expression,
            variation=variation,
        )

        generated_images = []
        for image_tensor in images:
            filename = f"{uuid.uuid4().hex}.png"
            filepath = os.path.join(GENERATED_FOLDER, filename)
            save_image(image_tensor, filepath)
            generated_images.append("/static/generated/" + filename)

        return jsonify({
            "success": True,
            "images": generated_images
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
