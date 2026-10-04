import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FFHQ_MODEL_PATH = os.path.join("models", "stylegan2-ffhq-config-f.pkl")


class DummyNetwork:
    def __setstate__(self, state):
        self.__dict__.update(state)


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "dnnlib" in module:
            return DummyNetwork
        return super().find_class(module, name)


class StyleGAN2GeneratorPyTorch(nn.Module):
    def __init__(self, mapping_vars, syn_vars, dlatent_avg, target_res=512):
        super().__init__()
        self.dlatent_avg = dlatent_avg.to(DEVICE)
        self.target_res = target_res

        # Load Mapping Network (8 FC layers)
        for i in range(8):
            w = torch.from_numpy(mapping_vars[f"Dense{i}/weight"]).float()
            b = torch.from_numpy(mapping_vars[f"Dense{i}/bias"]).float()
            w_scaled = w * ((2.0**0.5) / (512**0.5) * 0.01)
            b_scaled = b * 0.01
            self.register_buffer(f"map_w_{i}", w_scaled.to(DEVICE))
            self.register_buffer(f"map_b_{i}", b_scaled.to(DEVICE))

        # Constant 4x4 starting input
        const = torch.from_numpy(syn_vars["4x4/Const/const"]).float()
        self.register_buffer("const", const.to(DEVICE))

        # Store synthesis variables converted to PyTorch tensors on DEVICE
        self.syn_vars = {}
        for k, v in syn_vars.items():
            if isinstance(v, np.ndarray):
                self.syn_vars[k] = torch.from_numpy(v).float().to(DEVICE)
            else:
                self.syn_vars[k] = v

    def style_mapping(self, z, truncation=0.7):
        # PixelNorm
        x = z / torch.sqrt(torch.mean(z**2, dim=-1, keepdim=True) + 1e-8)
        for i in range(8):
            w = getattr(self, f"map_w_{i}")
            b = getattr(self, f"map_b_{i}")
            x = x @ w + b
            x = F.leaky_relu(x, 0.2)

        if truncation != 1.0:
            x = self.dlatent_avg.unsqueeze(0) + truncation * (
                x - self.dlatent_avg.unsqueeze(0)
            )
        return x

    def modulated_conv(self, x, w_latent, layer_prefix, is_torgb=False, upsample=False):
        w_conv = self.syn_vars[f"{layer_prefix}/weight"]
        b_conv = self.syn_vars[f"{layer_prefix}/bias"]
        w_mod = self.syn_vars[f"{layer_prefix}/mod_weight"]
        b_mod = self.syn_vars[f"{layer_prefix}/mod_bias"]

        # Affine style transform (w_mod shape: 512 x in_channels)
        style = w_latent @ w_mod * (1.0 / (512**0.5)) + b_mod

        if not is_torgb:
            w_conv_pt = w_conv.permute(3, 2, 0, 1)
            out_ch, in_ch, k1, k2 = w_conv_pt.shape
            w_scaled = w_conv_pt * (1.0 / ((in_ch * k1 * k2) ** 0.5))
        else:
            w_conv_pt = w_conv.permute(3, 2, 0, 1)
            out_ch, in_ch, k1, k2 = w_conv_pt.shape
            w_scaled = w_conv_pt * (1.0 / (in_ch**0.5))

        # Modulate & Demodulate (per sample in batch)
        s = style.view(1, 1, in_ch, 1, 1)
        w_modulated = w_scaled.unsqueeze(0) * s
        if not is_torgb:
            d = torch.rsqrt(
                torch.sum(w_modulated**2, dim=[2, 3, 4], keepdim=True) + 1e-8
            )
            w_demod = (w_modulated * d).squeeze(0)
        else:
            w_demod = w_modulated.squeeze(0)

        if upsample:
            x = F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )

        x = F.conv2d(x, w_demod, padding=1 if not is_torgb else 0)

        if not is_torgb and f"{layer_prefix}/noise_strength" in self.syn_vars:
            n_str = float(self.syn_vars[f"{layer_prefix}/noise_strength"])
            noise = torch.randn(
                x.shape[0], 1, x.shape[2], x.shape[3], device=x.device
            )
            x = x + noise * n_str

        x = x + b_conv.view(1, -1, 1, 1)

        if not is_torgb:
            x = F.leaky_relu(x, 0.2) * (2.0**0.5)

        return x

    def forward(
        self,
        z,
        truncation=0.7,
        skin_tone=0.5,
        face_age=0.5,
        expression=0.5,
        variation=0.5,
    ):
        w = self.style_mapping(z, truncation=truncation)

        # Apply subtle control latent manipulations in W-space
        w_offset = torch.zeros_like(w)
        w_offset[:, 0:128] += (skin_tone - 0.5) * 0.15
        w_offset[:, 128:256] += (face_age - 0.5) * 0.20
        w_offset[:, 256:384] += (expression - 0.5) * 0.25
        w_offset[:, 384:512] += (variation - 0.5) * 0.15
        w = w + w_offset

        # 4x4 Block
        x = self.const.repeat(z.shape[0], 1, 1, 1)
        x = self.modulated_conv(x, w, "4x4/Conv")
        rgb = self.modulated_conv(x, w, "4x4/ToRGB", is_torgb=True)

        res_blocks = [
            ("8x8", 8),
            ("16x16", 16),
            ("32x32", 32),
            ("64x64", 64),
            ("128x128", 128),
            ("256x256", 256),
            ("512x512", 512),
        ]

        for block_name, res in res_blocks:
            if res > self.target_res:
                break
            x = self.modulated_conv(x, w, f"{block_name}/Conv0_up", upsample=True)
            x = self.modulated_conv(x, w, f"{block_name}/Conv1")
            rgb = F.interpolate(
                rgb, scale_factor=2, mode="bilinear", align_corners=False
            ) + self.modulated_conv(x, w, f"{block_name}/ToRGB", is_torgb=True)

        return rgb


_model = None


def load_model():
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(FFHQ_MODEL_PATH):
        raise FileNotFoundError(
            f"Official NVIDIA StyleGAN2 FFHQ model checkpoint not found at {FFHQ_MODEL_PATH}."
        )

    print(f"Loading official StyleGAN2 FFHQ checkpoint from {FFHQ_MODEL_PATH}...")
    with open(FFHQ_MODEL_PATH, "rb") as f:
        obj = CustomUnpickler(f).load()

    Gs = obj[2]
    mapping_vars = dict(Gs.components["mapping"].variables)
    syn_vars = dict(Gs.components["synthesis"].variables)
    dlatent_avg = torch.from_numpy(dict(Gs.variables)["dlatent_avg"]).float()

    _model = StyleGAN2GeneratorPyTorch(
        mapping_vars, syn_vars, dlatent_avg, target_res=512
    ).to(DEVICE)
    _model.eval()
    print("StyleGAN2 FFHQ PyTorch Model successfully initialized!")
    return _model


@torch.no_grad()
def generate_faces(
    num_images=1,
    truncation=0.7,
    seed=None,
    skin_tone=0.5,
    face_age=0.5,
    expression=0.5,
    variation=0.5,
):
    model = load_model()

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    images = []
    for _ in range(num_images):
        z = torch.randn(1, 512, device=DEVICE)
        rgb = model(
            z,
            truncation=truncation,
            skin_tone=skin_tone,
            face_age=face_age,
            expression=expression,
            variation=variation,
        )
        images.append(rgb)

    return images


def save_image(tensor, path):
    tensor = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

    # Dynamic color range normalization to standard 0-255 RGB
    rgb_min = tensor.min()
    rgb_max = tensor.max()
    img_np = (tensor - rgb_min) / (rgb_max - rgb_min + 1e-8) * 255.0
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    image = Image.fromarray(img_np, mode="RGB")
    image.save(path)
    return image