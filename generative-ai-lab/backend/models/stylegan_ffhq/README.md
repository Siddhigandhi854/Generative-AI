---
library_name: stylegan-pytorch
license: mit
tags:
- gan
- image-generation
- model_hub_mixin
- pytorch_model_hub_mixin
- stylegan
- stylegan2
- unconditional-image-generation
- pytorch
---

### Generate Faces using the app : [https://stylegan-face-generation.streamlit.app/]


This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) integration:
- Code: https://github.com/HajarHAMDOUCH01/Face-Generator-StyleGAN-PyTorch
- Paper: [https://arxiv.org/abs/1812.04948]
- Paper: [https://arxiv.org/abs/1912.04958]
- Dataset(128*128 , 65k from FFHQ Dataset): [https://www.kaggle.com/datasets/hajarhamdouch/ffhq-128128-65k]

# StyleGAN Face Generator (128×128)

This model generates realistic human faces at 128×128 resolution using a StyleGAN1 architecture trained with StyleGAN2 regularization techniques.

## Model Description

- **Architecture**: StyleGAN 
- **Training**: Lazy R1 regularization + Path Length Regularization 
- **Dataset**: FFHQ (Flickr-Faces-HQ) 
- **Resolution**: 128×128 pixels
- **Latent dimension**: 512
- **Features**: Style mixing, noise injection, truncation trick

## Usage

### Quick Start

```python
import torch
from torchvision.utils import save_image
from huggingface_hub import hf_hub_download
import sys, os

# Download and load model
model_file = hf_hub_download(
    repo_id="hajar001/stylegan2-ffhq-128",
    filename="style_gan.py"
)
sys.path.insert(0, os.path.dirname(model_file))
from style_gan import StyleGAN

model = StyleGAN.from_pretrained("hajar001/stylegan2-ffhq-128")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Generate a single face
with torch.no_grad():
    z = torch.randn(1, 512, device=device)
    images = model.generate(z, truncation_psi=0.7)

# Denormalize from [-1, 1] to [0, 1]
images = (images + 1) / 2
images = torch.clamp(images, 0, 1)

save_image(images, "generated_face.png")
print("Generated face saved to generated_face.png")
```

### Generate Multiple Faces

```python
# Generate 16 faces in a 4×4 grid
with torch.no_grad():
    z = torch.randn(16, 512, device=device)
    images = model.generate(z, truncation_psi=0.7)

images = (images + 1) / 2
images = torch.clamp(images, 0, 1)

save_image(images, "generated_faces_grid.png", nrow=4)
print("Generated 16 faces")
```

### Truncation Trick

The `truncation_psi` parameter controls the trade-off between quality and diversity:
- `1.0`: Maximum diversity, lower quality
- `0.7`: Balanced (recommended)
- `0.5`: Higher quality, less diversity

```python
# High quality, less diverse
images = model.generate(z, truncation_psi=0.5)

# More diverse, slightly lower quality
images = model.generate(z, truncation_psi=1.0)
```

### Style Mixing

```python
# Generate two random latent codes
z1 = torch.randn(1, 512, device=device)
z2 = torch.randn(1, 512, device=device)

# Mix styles (coarse features from z1, fine details from z2)
with torch.no_grad():
    w1 = model.mapping(z1)
    w2 = model.mapping(z2)
    
    # Create mixed w: first 4 layers from w1, rest from w2
    w_mixed = torch.cat([
        w1.unsqueeze(1).expand(-1, 4, -1),
        w2.unsqueeze(1).expand(-1, 8, -1)
    ], dim=1)
    
    mixed_image = model.synthesis(w_mixed)

mixed_image = (mixed_image + 1) / 2
save_image(mixed_image, "style_mixed.png")
```

## Training Details

- **Epochs**: 38+ (ongoing training)
- **Optimizer**: Adam (β₁=0.0, β₂=0.99)
- **Learning Rates**: G=0.0002, D=0.0002
- **Regularization**: 
  - R1 gamma: 5.0 (applied every 16 iterations)
  - Path Length Regularization: 1.0 (applied every 16 iterations)
- **Batch Size**: 32
- **Style Mixing Probability**: 0.8

## Limitations

- Resolution limited to 128×128

## Technical Notes

This is a **hybrid approach**: StyleGAN1 architecture (AdaIN-based style modulation) trained with StyleGAN2 regularization techniques (R1 + PLR). This combines the simplicity of StyleGAN1 with the improved training stability of StyleGAN2.

## Citation

If you use this model, please cite the original StyleGAN papers:

```bibtex
@article{karras2018stylebased,
  title={A Style-Based Generator Architecture for Generative Adversarial Networks},
  author={Karras, Tero and Laine, Samuli and Aila, Timo},
  journal={arXiv preprint arXiv:1812.04948},
  year={2018}
}

@article{karras2019stylegan2,
  title={Analyzing and Improving the Image Quality of StyleGAN},
  author={Karras, Tero and Laine, Samuli and Aittala, Miika and Hellsten, Janne and Lehtinen, Jaakko and Aila, Timo},
  journal={arXiv preprint arXiv:1912.04958},
  year={2019}
}
```

## Resources

- **Code Repository**: https://github.com/HajarHAMDOUCH01/Face-Generator-StyleGAN-PyTorch
- **StyleGAN Paper**: https://arxiv.org/abs/1812.04948
- **StyleGAN2 Paper**: https://arxiv.org/abs/1912.04958
- **FFHQ Dataset**: https://github.com/NVlabs/ffhq-dataset

## License

MIT License - See LICENSE file for details