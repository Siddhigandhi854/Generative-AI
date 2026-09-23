# GenAI VAE Latent Space Explorer

## Project structure

genai_vae_miniproject/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
└── models/
    └── vae_decoder.pkl   # REQUIRED trained model file

## Run

1. Create/activate a Python virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Put the trained `vae_decoder.pkl` inside:

   models/vae_decoder.pkl

4. Start:

   python app.py

5. Open:

   http://127.0.0.1:5000

## Important

The original uploaded files did not include `models/vae_decoder.pkl`.
The application loads that trained decoder at startup, so image generation
cannot work until the actual trained model file is supplied.
