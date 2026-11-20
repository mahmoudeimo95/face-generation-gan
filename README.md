# Realistic Human Face Generation using GAN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahmoudeimo95/face-generation-gan/blob/main/face_generation_gan.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DCGAN-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

A Generative Adversarial Network (DCGAN) that generates realistic synthetic human faces from random noise, trained on the CelebA dataset. Includes a webcam demo that shows the GAN's closest version of your own face.

## Objectives

- Understand how GANs generate new realistic images
- Build a **Generator** that creates synthetic faces
- Build a **Discriminator** that classifies real vs. fake images
- Train both models with adversarial learning
- Evaluate generated face quality throughout training

## Results

**Final generated faces**

![Final generated faces](results/final_faces.png)

**Progress over training epochs**

![Progress comparison](results/progress_comparison.png)

**Training losses**

![Loss curves](results/loss_curves.png)

## Pipeline

1. **Data:** CelebA ([Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)), center-cropped to 148x148, resized to 64x64, normalized to [-1, 1]
2. **Generator:** 100-dim latent vector, then 5 transposed-conv layers (BatchNorm + ReLU), `tanh` output, giving a 3x64x64 image
3. **Discriminator:** 5 conv layers (BatchNorm + LeakyReLU), outputs a real/fake logit
4. **Training:** alternating Discriminator / Generator updates on random latent vectors
5. **Monitoring:** losses, D(x) / D(G(z)) scores, and a sample grid saved every epoch from a fixed noise batch
6. **Evaluation:** epoch-by-epoch visual comparison, real vs. generated faces, latent-space interpolation

### Hyperparameters

| Setting | Value |
|---|---|
| Image size | 64x64 |
| Latent dimension | 100 |
| Batch size | 128 |
| Epochs | 30 |
| Optimizer | Adam (lr = 2e-4, beta1 = 0.5) |
| Loss | BCE with logits |
| Label smoothing | Real label = 0.9 |
| Training images | 100,000 (configurable) |

## Webcam Demo

The last section of the notebook launches a Gradio app. Take a webcam photo and the app will:

1. Detect and crop your face
2. Search the Generator's latent space for the noise vector that best recreates it (GAN inversion)
3. Show your face next to the GAN's closest version, plus 16 new random faces

The result is an approximate likeness, since a small 64x64 DCGAN cannot reproduce a specific person exactly.

## Getting Started

**Google Colab (recommended):** click the Colab badge above, select a GPU runtime (Runtime > Change runtime type > T4 GPU), and run all cells.

**Local:**

```bash
git clone https://github.com/mahmoudeimo95/face-generation-gan.git
cd face-generation-gan
pip install -r requirements.txt
jupyter notebook face_generation_gan.ipynb
```

The dataset downloads automatically through `kagglehub` (a Kaggle account may be required).

## Project Structure

```
face-generation-gan/
├── face_generation_gan.ipynb   # full pipeline: data, models, training, evaluation, webcam demo
├── requirements.txt
├── results/                    # final figures used in this README
├── LICENSE
└── README.md
```

## Possible Improvements

- Train longer on the full 202k images
- Try WGAN-GP or spectral normalization for more stable training
- Increase resolution to 128x128
- Add quantitative evaluation (FID)

## Author

**Mahmoud Atef** · [GitHub](https://github.com/mahmoudeimo95)

## License

Released under the [MIT License](LICENSE).
