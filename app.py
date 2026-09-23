from flask import Flask, render_template, request, jsonify
from model import generate_text

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()

    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({
            "error": "Please enter a prompt."
        }), 400

    try:
        generated_text = generate_text(prompt)

        return jsonify({
            "text": generated_text
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)