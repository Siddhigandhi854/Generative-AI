import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

class DummyNetwork:
    def __setstate__(self, state):
        self.__dict__.update(state)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if 'dnnlib' in module:
            return DummyNetwork
        return super().find_class(module, name)

def load_stylegan2_weights(pkl_path):
    with open(pkl_path, 'rb') as f:
        obj = CustomUnpickler(f).load()
    Gs = obj[2]
    mapping_vars = dict(Gs.components['mapping'].variables)
    syn_vars = dict(Gs.components['synthesis'].variables)
    dlatent_avg = torch.from_numpy(dict(Gs.variables)['dlatent_avg']).float()
    return mapping_vars, syn_vars, dlatent_avg

class StyleGAN2GeneratorPyTorch(nn.Module):
    def __init__(self, mapping_vars, syn_vars, dlatent_avg, target_res=512):
        super().__init__()
        self.dlatent_avg = dlatent_avg
        self.target_res = target_res
        
        # Load Mapping Network
        self.mapping_weights = []
        self.mapping_biases = []
        for i in range(8):
            w = torch.from_numpy(mapping_vars[f'Dense{i}/weight']).float()
            b = torch.from_numpy(mapping_vars[f'Dense{i}/bias']).float()
            w_scaled = w * ((2.0**0.5) / (512**0.5) * 0.01)
            b_scaled = b * 0.01
            self.register_buffer(f'map_w_{i}', w_scaled)
            self.register_buffer(f'map_b_{i}', b_scaled)

        # Constant 4x4 input
        const = torch.from_numpy(syn_vars['4x4/Const/const']).float()
        self.register_buffer('const', const)

        # Store syn_vars for dynamic forward pass
        self.syn_vars = syn_vars

    def style_mapping(self, z, truncation=0.7):
        # PixelNorm
        x = z / torch.sqrt(torch.mean(z**2, dim=-1, keepdim=True) + 1e-8)
        for i in range(8):
            w = getattr(self, f'map_w_{i}')
            b = getattr(self, f'map_b_{i}')
            x = x @ w + b
            x = F.leaky_relu(x, 0.2)
        
        if truncation != 1.0:
            x = self.dlatent_avg.unsqueeze(0) + truncation * (x - self.dlatent_avg.unsqueeze(0))
        return x

    def modulated_conv(self, x, w_latent, layer_prefix, is_torgb=False, upsample=False):
        w_conv = torch.from_numpy(self.syn_vars[f'{layer_prefix}/weight']).float()
        b_conv = torch.from_numpy(self.syn_vars[f'{layer_prefix}/bias']).float()
        w_mod = torch.from_numpy(self.syn_vars[f'{layer_prefix}/mod_weight']).float()
        b_mod = torch.from_numpy(self.syn_vars[f'{layer_prefix}/mod_bias']).float()
        
        # Affine style transform
        # mod_weight shape: (512, in_ch)
        style = w_latent @ w_mod * (1.0 / (512**0.5)) + b_mod # (batch, channels)

        if not is_torgb:
            # weight shape in TF: (kernel, kernel, in_ch, out_ch) e.g. (3, 3, 512, 512)
            # convert to PyTorch: (out_ch, in_ch, kernel, kernel)
            w_conv = w_conv.permute(3, 2, 0, 1)
            out_ch, in_ch, k1, k2 = w_conv.shape
            w_scaled = w_conv * (1.0 / ((in_ch * k1 * k2)**0.5))
        else:
            # torgb weight shape in TF: (1, 1, in_ch, 3)
            # convert to PyTorch: (3, in_ch, 1, 1)
            w_conv = w_conv.permute(3, 2, 0, 1)
            out_ch, in_ch, k1, k2 = w_conv.shape
            w_scaled = w_conv * (1.0 / (in_ch**0.5))

        # Modulate & Demodulate (for batch_size = 1)
        # style shape: (1, in_ch)
        s = style.view(1, 1, in_ch, 1, 1)
        w_modulated = w_scaled.unsqueeze(0) * s # (1, out_ch, in_ch, k1, k2)
        if not is_torgb:
            d = torch.rsqrt(torch.sum(w_modulated**2, dim=[2, 3, 4], keepdim=True) + 1e-8)
            w_demod = (w_modulated * d).squeeze(0)
        else:
            w_demod = w_modulated.squeeze(0)

        if upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = F.conv2d(x, w_demod, padding=1 if not is_torgb else 0)

        if not is_torgb and f'{layer_prefix}/noise_strength' in self.syn_vars:
            n_str = float(self.syn_vars[f'{layer_prefix}/noise_strength'])
            noise_idx = layer_prefix # e.g. 4x4/Conv
            # generate noise
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
            x = x + noise * n_str

        x = x + b_conv.view(1, -1, 1, 1)

        if not is_torgb:
            x = F.leaky_relu(x, 0.2) * (2.0**0.5)

        return x

    def forward(self, z, truncation=0.7):
        w = self.style_mapping(z, truncation=truncation)
        
        # 4x4 Block
        x = self.const.repeat(z.shape[0], 1, 1, 1)
        x = self.modulated_conv(x, w, '4x4/Conv')
        rgb = self.modulated_conv(x, w, '4x4/ToRGB', is_torgb=True)

        res_blocks = [
            ('8x8', 8),
            ('16x16', 16),
            ('32x32', 32),
            ('64x64', 64),
            ('128x128', 128),
            ('256x256', 256),
            ('512x512', 512),
            ('1024x1024', 1024),
        ]

        for block_name, res in res_blocks:
            if res > self.target_res:
                break
            x = self.modulated_conv(x, w, f'{block_name}/Conv0_up', upsample=True)
            x = self.modulated_conv(x, w, f'{block_name}/Conv1')
            rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False) + \
                  self.modulated_conv(x, w, f'{block_name}/ToRGB', is_torgb=True)

        return rgb

print('Loading model weights...')
mapping_vars, syn_vars, dlatent_avg = load_stylegan2_weights('models/stylegan2-ffhq-config-f.pkl')
gen = StyleGAN2GeneratorPyTorch(mapping_vars, syn_vars, dlatent_avg, target_res=512)

z = torch.randn(1, 512)
with torch.no_grad():
    out_rgb = gen(z, truncation=0.7)

print('Output RGB shape:', out_rgb.shape, 'min:', out_rgb.min().item(), 'max:', out_rgb.max().item())

# Save image with dynamic contrast normalization
rgb = out_rgb.squeeze(0).permute(1, 2, 0).numpy()
rgb_min = rgb.min()
rgb_max = rgb.max()
img_np = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8) * 255.0
img_np = np.clip(img_np, 0, 255).astype(np.uint8)
img = Image.fromarray(img_np)
os.makedirs('scratch', exist_ok=True)
img.save('scratch/test_face.png')
print('Saved scratch/test_face.png successfully!')
