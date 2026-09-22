# Conditional Diffusion Generator

A class-conditional diffusion model for generating synthetic industrial surface textures using **Denoising Diffusion Probabilistic Models (DDPM)**. The project uses a U-Net noise-prediction architecture with **Classifier-Free Guidance (CFG)** and supports accelerated **DDIM sampling**.

## Overview

This project focuses on generating realistic synthetic textures conditioned on an industrial object category. It progresses from diffusion experiments on simpler datasets to real industrial texture generation using **MVTec AD**.

The final model generates **64×64 RGB images** from selected MVTec categories and provides multiple sampling methods for comparing generation quality and speed.

## Key Features

* Class-conditional DDPM implemented in **PyTorch**
* U-Net based noise prediction architecture
* Sinusoidal timestep embeddings
* Learned class embeddings for conditional generation
* Residual blocks with **GroupNorm + SiLU**
* Self-attention at low spatial resolutions
* **Classifier-Free Guidance (CFG)**
* **EMA** model weights for sampling
* Standard **DDPM sampling**
* Accelerated **DDIM sampling**
* FID-based generation quality evaluation
* Sampling speed benchmarking
* Interactive **Gradio demo**
* Denoising trajectory visualization

## Dataset

The final stage uses the **MVTec AD** dataset and focuses on defect-free images from the `train/good` directories.

The project also includes an earlier texture-generation stage using the **Describable Textures Dataset (DTD)** before moving to MVTec AD.

### MVTec AD

* RGB images
* 64×64 training resolution
* Category-conditioned generation
* Normal/defect-free samples used for training

## Model Architecture

The generator uses a U-Net architecture that predicts the noise added to an image during the diffusion process.

### Main Components

```text
Input Image
     │
     ├── Timestep Embedding
     │
     └── Class Embedding
             │
             ▼
      Conditional U-Net
             │
     ┌───────┴───────┐
     │               │
 Downsampling    Upsampling
     │               │
     └── Self-Attention
             │
             ▼
       Noise Prediction
             │
             ▼
      Diffusion Sampling
             │
             ▼
     Generated Texture
```

The model uses:

* Sinusoidal time embeddings
* Class embeddings
* Residual blocks
* Group Normalization
* SiLU activations
* Skip connections
* Self-attention
* RGB input/output

## Diffusion Process

The model follows the standard DDPM framework with a **1000-step linear noise schedule**.

During training, noise is progressively added to real images. The U-Net learns to predict this noise so that the reverse diffusion process can reconstruct an image from random noise.

For conditional generation, the model receives both:

* Diffusion timestep `t`
* Target class label `y`

## Classifier-Free Guidance

Classifier-Free Guidance allows the model to control how strongly the generated image follows the selected class.

During training, class conditioning is randomly dropped for some samples. During generation, conditional and unconditional predictions are combined:

```text
ε_guided = ε_uncond + s(ε_cond − ε_uncond)
```

where `s` is the guidance scale.

This allows the same model to generate images with different levels of adherence to the selected category.

## DDPM and DDIM Sampling

The project supports two sampling approaches:

### DDPM

Uses the full diffusion trajectory with 1000 sampling steps.

* Higher computational cost
* Standard DDPM sampling procedure
* Useful as a reference for generation quality

### DDIM

Uses a reduced number of diffusion steps while keeping the trained model unchanged.

The notebook evaluates DDIM with:

* 20 steps
* 50 steps
* 100 steps

This allows generation speed and quality to be compared against the full DDPM process.

## Evaluation

The project includes benchmarking for both **generation quality** and **sampling speed**.

### FID

Fréchet Inception Distance (FID) is calculated by comparing generated samples against real MVTec reference images.

The benchmark is configured to use a large sample set so that the FID estimate is more meaningful.

### Speed

The notebook measures:

* Total generation time
* Number of generated images
* Seconds per image

for different sampling methods.

## Interactive Demo

A Gradio interface is included for interactive generation.

The user can select:

* MVTec texture category
* Guidance scale
* Sampling method

The demo generates:

1. A final synthetic texture
2. A visualization of the denoising process

Supported samplers include:

```text
DDPM — 1000 steps
DDIM — 50 steps
DDIM — 100 steps
```

## Project Structure

```text
Conditional-Diffusion-Generator/
│
├── phase_c_ddpm_texture_fixed.ipynb
│   └── Texture training with DTD and MVTec AD
│
├── phase_d_standalone_mvtec.ipynb
│   └── MVTec generation, evaluation and Gradio demo
│
└── README.md
```

## Requirements

The notebooks are designed to run in **Google Colab** with GPU support.

Main libraries include:

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Pillow
* Gradio
* torch-fidelity

## How to Run

### 1. Train the model

Open the Phase C notebook and run the training pipeline using a GPU runtime.

The notebook saves checkpoints to Google Drive so training can be resumed.

### 2. Run MVTec generation

Open the Phase D notebook after training.

The notebook loads the trained checkpoint and provides:

* MVTec sample generation
* DDPM/DDIM comparison
* Speed benchmarking
* FID evaluation
* Gradio demonstration

### 3. Launch the Demo

Run the Gradio section to launch the interactive conditional texture generator.

## Applications

Conditional industrial texture generation can be useful for:

* Synthetic dataset augmentation
* Industrial computer vision
* Anomaly detection research
* Defect classification
* Visual inspection systems
* Generative modeling of manufacturing surfaces

## References

* Ho et al. (2020), *Denoising Diffusion Probabilistic Models*
* Song et al. (2021), *Denoising Diffusion Implicit Models*
* Ho & Salimans (2022), *Classifier-Free Diffusion Guidance*

---

**Project:** Conditional Diffusion Generator
**Framework:** PyTorch
**Domain:** Generative AI / Diffusion Models / Industrial Computer Vision
