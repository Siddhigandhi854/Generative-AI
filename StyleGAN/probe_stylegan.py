import torch
from model import load_model, generate_faces, save_image

print("Testing StyleGAN2 FFHQ PyTorch Model probe...")
model = load_model()
print("Model loaded successfully!")

images = generate_faces(num_images=1, truncation=0.7, seed=42)
print("Generated tensor count:", len(images))
print("Image tensor shape:", images[0].shape)

saved_img = save_image(images[0], "static/probe_sample.png")
print("Saved static/probe_sample.png with size:", saved_img.size)
