```markdown
# Build → Break → Improve: Navigating Synthetic Reality

> Synthetic Image Detector · Adversarial Evasion · Robustness Defense  
> Dataset: CIFAKE | Model: ResNet-18 Transfer Learning | Framework: PyTorch

---

## Table of Contents

1. [Problem Overview](#1-problem-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Requirements](#4-requirements)
5. [Setup & Installation](#5-setup--installation)
6. [How to Run](#6-how-to-run)
7. [Output Files](#7-output-files)
8. [Phase 1 – Build](#8-phase-1--build-synthetic-image-detector)
9. [Phase 2 – Break](#9-phase-2--break-adversarial-evasion)
10. [Phase 3 – Improve](#10-phase-3--improve-robustness-defense)
11. [Final Results](#11-final-results)
12. [Key Takeaways](#12-key-takeaways)
13. [References](#13-references)

---

## 1. Problem Overview

Modern generative AI models produce synthetic images that are nearly
indistinguishable from real photographs, creating serious risks in
cybersecurity, misinformation detection, and digital forensics.

This project follows a 3-phase research-inspired cycle:

| Phase | Name | Goal |
|-------|------|------|
| 1 | **Build** | Train a binary classifier: REAL vs AI-generated (FAKE) |
| 2 | **Break** | Craft adversarial modifications that fool the detector |
| 3 | **Improve** | Design and prototype a defense based on found weaknesses |

> This mirrors how real-world AI security tools evolve — build, get attacked,
> diagnose, harden.

---

## 2. Dataset

### CIFAKE – Real and AI-Generated Synthetic Images

| Property       | Value                                         |
|----------------|-----------------------------------------------|
| Source         | Kaggle                                        |
| Total Images   | 120,000 (60,000 REAL + 60,000 FAKE)           |
| Resolution     | 32×32 RGB (resized to 64×64 during training)  |
| REAL source    | CIFAR-10 original photographs                 |
| FAKE source    | Stable Diffusion generated equivalents        |
| License        | CC0 Public Domain                             |
| Kaggle URL     | https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images |

---

### How to Download

**Option A — Kaggle CLI (Recommended):**

```bash
# Step 1: Install Kaggle CLI
pip install kaggle

# Step 2: Place API key at:
#   Windows : C:\Users\<YourUsername>\.kaggle\kaggle.json
#   Linux   : ~/.kaggle/kaggle.json
# Get your key from: https://www.kaggle.com/settings → API → Create New Token

# Step 3: Download dataset
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images

# Step 4: Extract (Windows PowerShell)
Expand-Archive -Path "cifake-real-and-ai-generated-synthetic-images.zip" `
               -DestinationPath "data"

# Step 4: Extract (Linux / macOS)
unzip cifake-real-and-ai-generated-synthetic-images.zip -d data
```

**Option B — Manual Download:**
1. Visit: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
2. Click **Download** (requires Kaggle account)
3. Extract the zip into the `data/` folder

---

### Expected Folder Structure After Extraction

```
data/
├── train/
│   ├── REAL/     ← 50,000 images (.jpg)
│   └── FAKE/     ← 50,000 images (.jpg)
└── test/
    ├── REAL/     ← 10,000 images (.jpg)
    └── FAKE/     ← 10,000 images (.jpg)
```

> ⚠️ `data/` is listed in `.gitignore` and is **NOT included** in this repo.
> You must download the dataset separately before running any phase.

---

## 3. Project Structure

```
cifake-detector/
│
├── data/                            ← NOT in repo (download from Kaggle)
│   ├── train/
│   │   ├── REAL/
│   │   └── FAKE/
│   └── test/
│       ├── REAL/
│       └── FAKE/
│
├── models/                          ← NOT in repo (auto-created after training)
│   ├── best.pth                     # Phase 1: trained ResNet-18 detector
│   ├── robust.pth                   # Phase 3: adversarially hardened model
│   ├── pgd_train_imgs.pt            # Cached PGD adversarial training images
│   ├── pgd_train_lbls.pt
│   ├── blur_train_imgs.pt           # Cached Blur augmented training images
│   ├── blur_train_lbls.pt
│   ├── adv_test_imgs.pt             # Cached PGD adversarial test images
│   ├── adv_test_lbls.pt
│   ├── blur_test_imgs.pt            # Cached Blur adversarial test images
│   └── blur_test_lbls.pt
│
├── outputs/                         ← NOT in repo (auto-created after running)
│   ├── training_samples.png
│   ├── test_samples.png
│   ├── confusion_matrix.png
│   ├── gradcam_fake.png
│   ├── gradcam_real.png
│   ├── saliency_fake.png
│   ├── saliency_real.png
│   ├── phase2/
│   │   ├── targets.png
│   │   ├── before_after.png
│   │   ├── confidence_trajectories.png
│   │   ├── gradcam_comparison.png
│   │   ├── fft_analysis.png
│   │   └── attack_summary.png
│   └── phase3/
│       ├── finetune_curves.png
│       ├── robustness_comparison.png
│       ├── confusion_matrices.png
│       └── adversarial_predictions.png
│
├── phase1.py                        # Train the detector (Build)
├── phase1_eval.py                   # Evaluate model + generate explainability
├── phase2.py                        # Adversarial evasion attacks (Break)
├── phase3.py                        # Robustness defense + re-evaluation (Improve)
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Excludes data/, models/, outputs/, venv/
└── README.md                        # This file
```

---

## 4. Requirements

**`requirements.txt`:**

```
torch
torchvision
scikit-learn
matplotlib
tqdm
Pillow
opencv-python
numpy
```

### Tested Environment

| Component   | Version           |
|-------------|-------------------|
| Python      | 3.11              |
| PyTorch     | 2.x (CPU)         |
| Torchvision | 0.x compatible    |
| OS          | Windows 11        |

### GPU Installation (Optional — 10× faster training)

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 5. Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/cifake-detector.git
cd cifake-detector

# 2. Create a virtual environment
python -m venv venv

# Activate — Windows:
venv\Scripts\activate

# Activate — Linux / macOS:
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Download and extract the CIFAKE dataset into data/
#    (see Section 2 above)

# 5. Verify your data path
python -c "import os; print(os.path.exists('data/train/REAL'))"
# Should print: True
```

---

## 6. How to Run

> Run each phase **in order**. Each phase depends on outputs from the previous one.

---

### Phase 1 — Train the Detector

```bash
python phase1.py
```

**What it does:**
- Loads CIFAKE train/val/test splits
- Downloads ResNet-18 pretrained weights (~45MB, one-time)
- Stage 1: trains classifier head only (5 epochs, frozen backbone)
- Stage 2: fine-tunes full network (5 epochs, low LR)
- Saves best checkpoint to `models/best.pth`
- Saves training curve plots to `outputs/`

> ⏱ **Estimated time (CPU):** 2–3 hours for full 85k training set  
> 💡 **Speed tip:** Add these two lines after the split in `phase1.py` to use 20% of data:
> ```python
> train_idx = train_idx[:int(len(train_idx) * 0.2)]
> val_idx   = val_idx[:int(len(val_idx)   * 0.2)]
> ```

---

### Phase 1 Eval — Evaluate + Explainability

```bash
python phase1_eval.py
```

**What it does:**
- Loads saved `models/best.pth` (no training)
- Runs inference on full 20k test set
- Prints Accuracy, Precision, Recall, F1, Classification Report
- Generates Grad-CAM heatmaps (FAKE + REAL)
- Generates Saliency Maps (FAKE + REAL)
- Saves confusion matrix and all visualizations to `outputs/`

> ⏱ **Estimated time (CPU):** ~5 minutes

---

### Phase 2 — Adversarial Attacks

```bash
python phase2.py
```

**What it does:**
- Selects 10 high-confidence FAKE targets from test set (P(fake) ≥ 0.90)
- Runs 5 attack types on each target:
  - Gaussian Blur (k = 1→15)
  - JPEG Compression (quality = 20)
  - FGSM (ε = 0.005→0.10)
  - PGD (ε = 0.03 and 0.05, 40–50 iterations)
  - Blur + PGD (combined)
- Generates confidence trajectory plots per attack
- Generates Grad-CAM before/after comparison
- Generates FFT frequency spectrum analysis
- Produces full attack success summary
- Saves all outputs to `outputs/phase2/`

> ⏱ **Estimated time (CPU):** ~20–30 minutes

---

### Phase 3 — Robustness Defense

```bash
python phase3.py
```

**What it does:**
- Generates 1000 PGD adversarial training examples (FAKE → still FAKE)
- Generates 1000 Blur-augmented training examples (FAKE → still FAKE)
- **Caches** all generated data to `models/*.pt` — re-runs load from cache instantly
- Builds combined training set (5k original + 1k PGD + 1k Blur = 7k total)
- Fine-tunes original model for 3 epochs (LR = 5e-5)
- Evaluates both original and robust model on:
  - Clean test set (20,000 images)
  - PGD adversarial test set (200 images)
  - Blur adversarial test set (200 images)
- Saves all comparison plots to `outputs/phase3/`

> ⏱ **Estimated time (CPU):**
> - First run: ~40–50 minutes (includes PGD generation)
> - Re-run: ~15 minutes (loads from cache)

---

## 7. Output Files

### Phase 1 (`outputs/`)

| File | Description |
|------|-------------|
| `training_samples.png` | Grid of sample images from training set |
| `test_samples.png` | Grid of sample images from test set |
| `confusion_matrix.png` | 2×2 confusion matrix on test set |
| `gradcam_fake.png` | Grad-CAM heatmaps for 6 FAKE test images |
| `gradcam_real.png` | Grad-CAM heatmaps for 6 REAL test images |
| `saliency_fake.png` | Saliency maps for 6 FAKE test images |
| `saliency_real.png` | Saliency maps for 6 REAL test images |
| `stage_1_head_only.png` | Loss + accuracy curves — Stage 1 training |
| `stage_2_fine-tuning.png` | Loss + accuracy curves — Stage 2 fine-tuning |

### Phase 2 (`outputs/phase2/`)

| File | Description |
|------|-------------|
| `targets.png` | 10 selected high-confidence FAKE targets |
| `before_after.png` | Original → 4 attacks with confidence per target |
| `confidence_trajectories.png` | P(fake) vs attack strength (Blur/FGSM/PGD) |
| `gradcam_comparison.png` | Attention shift: original vs evaded image |
| `fft_analysis.png` | FFT frequency spectrum before and after attack |
| `attack_summary.png` | Evasion rate + avg confidence drop per attack |

### Phase 3 (`outputs/phase3/`)

| File | Description |
|------|-------------|
| `finetune_curves.png` | Loss/accuracy during adversarial fine-tuning |
| `robustness_comparison.png` | Bar chart: original vs robust across all conditions |
| `confusion_matrices.png` | Side-by-side confusion matrices (clean test set) |
| `adversarial_predictions.png` | PGD examples: original vs robust predictions |

---

## 8. Phase 1 – Build: Synthetic Image Detector

### Model Architecture

```
Input: 64×64 RGB image (normalized mean=0.5, std=0.5)
   ↓
ResNet-18 Backbone (ImageNet pretrained)
   ↓
Global Average Pooling → 512-dim feature vector
   ↓
Dropout(0.3) → Linear(512→256) → ReLU → Dropout(0.2) → Linear(256→2)
   ↓
Output: [P(REAL), P(FAKE)]
```

### Training Strategy

| Stage | Backbone | Head | Epochs | LR |
|-------|----------|------|--------|----|
| 1 | Frozen | Trainable | 5 | 1e-3 |
| 2 | Trainable | Trainable | 5 | backbone: 1e-4, head: 1e-3 |

- **Early stopping:** patience = 3, based on validation F1
- **Scheduler:** StepLR, step=3, gamma=0.5

### Test Set Results

| Metric | Value |
|--------|-------|
| Accuracy | **97.11%** |
| Precision (FAKE) | 0.9758 |
| Recall (FAKE) | 0.9663 |
| F1-score (FAKE) | **0.9710** |

### Explainability Findings

From Grad-CAM and saliency maps:
- The model focuses heavily on **high-frequency edge and texture patterns**
  in FAKE images rather than semantic content (shapes, objects).
- REAL image attention is more diffuse and object-centered.
- This confirms the model learned **HF artifact fingerprints of generative models**
  — a significant latent vulnerability.

---

## 9. Phase 2 – Break: Adversarial Evasion

### Target Selection

10 FAKE test images with **P(fake) = 1.0000** selected as attack targets.

### Attack Results

| Attack | Evaded / 10 | Evasion Rate | Avg ΔP(fake) |
|--------|:-----------:|:------------:|:------------:|
| Blur (k=15) | 9 / 10 | 90.0% | 0.7162 |
| JPEG (q=20) | 0 / 10 | 0.0% | 0.0000 |
| FGSM (ε=0.05) | 0 / 10 | 0.0% | 0.0202 |
| FGSM (ε=0.10) | 0 / 10 | 0.0% | ~0.000 |
| **PGD (ε=0.03)** | **10 / 10** | **100.0%** | **1.0000** |
| **PGD (ε=0.05)** | **10 / 10** | **100.0%** | **1.0000** |
| **Blur + PGD** | **10 / 10** | **100.0%** | **1.0000** |

### Why the Attacks Worked

| Attack | Mechanism | Evidence |
|--------|-----------|----------|
| **Blur** | Removes HF noise patterns (GAN/diffusion fingerprints) | FFT outer rings fade after blurring |
| **PGD** | Gradient-based: directly optimizes away the FAKE signal | Grad-CAM attention shifts to irrelevant regions |
| **Blur+PGD** | Blur removes HF artifacts; PGD erases remaining features | Most realistic adversarial images |
| **JPEG/FGSM** | Block artifacts don't match detector's cues; single-step too coarse | Consistently fail across all targets |

**Core finding:**
> The detector is HF-dependent and non-robust. It detects synthetic images based on
> superficial noise patterns, not semantic image understanding.

---

## 10. Phase 3 – Improve: Robustness Defense

### Vulnerability Diagnosed

| Attack | Before Defense | Root Cause |
|--------|:--------------:|------------|
| PGD | 0% accuracy | Model relies on gradient-exploitable HF features |
| Blur | 27% accuracy | Model relies on HF artifact presence |

### Defense: Adversarial Fine-tuning

**Strategy directly derived from Phase 2 findings:**

```
Step 1: Generate 1000 PGD-perturbed FAKE training images
        → Label them STILL FAKE
        → Model learns to detect FAKEs even after gradient attack

Step 2: Generate 1000 Blur-augmented FAKE training images  
        → Label them STILL FAKE
        → Model learns to detect FAKEs without HF artifacts

Step 3: Mix with 5000 original clean training samples
        → Total: 7000 examples (clean data stays dominant)

Step 4: Fine-tune original model — 3 epochs, LR=5e-5
        → Low LR preserves clean accuracy
        → New adversarial samples build robustness
```

### Results After Defense

| Condition | Original Model | Robust Model | Improvement |
|-----------|:--------------:|:------------:|:-----------:|
| Clean Test Set | 97.11% | **97.16%** | +0.04% |
| PGD Adversarial | 0.00% | **100.00%** | **+100.00%** |
| Blur Adversarial | 27.00% | **100.00%** | **+73.00%** |

### Why This Works

Adversarial training forces the model to find **deeper, more stable features**.
When it sees PGD-perturbed or blurred FAKE images that look nearly real, it can no
longer rely on HF noise — it must learn lower-frequency semantic cues
(color distributions, spatial coherence, object statistics).
Clean accuracy is preserved because clean data remains the majority of training.

---

## 11. Final Results

```
╔══════════════════════════════════════════════════════════╗
║  PHASE     TASK                       KEY METRIC         ║
╠══════════════════════════════════════════════════════════╣
║  Phase 1   Build Detector             Accuracy : 97.11%  ║
║                                       F1 Score : 0.9710  ║
╠══════════════════════════════════════════════════════════╣
║  Phase 2   PGD Attack Evasion Rate    100%               ║
║            Blur Attack Evasion Rate    90%               ║
╠══════════════════════════════════════════════════════════╣
║  Phase 3   Robust Model (Clean)       Accuracy : 97.16%  ║
║            Robust Model (PGD)         Accuracy : 100%    ║
║            Robust Model (Blur)        Accuracy : 100%    ║
╚══════════════════════════════════════════════════════════╝
```

---

## 12. Key Takeaways

- **ResNet-18 + transfer learning** achieves 97% accuracy on CIFAKE
  with only ~3 hours of CPU training.

- The detector is **critically vulnerable to PGD (100% evasion) and
  Gaussian blur (90% evasion)** — confirming it learned HF generative
  artifacts rather than true semantic content differences.

- **Only 3 fine-tuning epochs on 7000 samples** are sufficient to completely
  eliminate both vulnerabilities while preserving baseline accuracy.

- **Adversarial training is directly interpretable here** — because we know
  exactly which features the model relied on (HF artifacts), we can design
  training examples that specifically remove that reliance.

- This workflow — build → attack → diagnose → harden — is identical to how
  production deepfake detectors and forensic AI tools are developed and maintained.

---

## 13. References

1. **CIFAKE Dataset:**  
   Bird, J. J. & Lotfi, A. (2023). *CIFAKE: Image Classification and Explainable
   Identification of AI-Generated Synthetic Images.* IEEE Access.  
   https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

2. **ResNet:**  
   He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.

3. **Grad-CAM:**  
   Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep
   Networks via Gradient-Based Localization.* ICCV.

4. **PGD Attack:**  
   Madry, A. et al. (2018). *Towards Deep Learning Models Resistant to
   Adversarial Attacks.* ICLR.

5. **Adversarial Training:**  
   Goodfellow, I. et al. (2015). *Explaining and Harnessing Adversarial
   Examples.* ICLR.

6. **FGSM:**  
   Goodfellow, I. et al. (2015). *Explaining and Harnessing Adversarial
   Examples.* ICLR.

---

> **Submission includes:** `phase1.py` · `phase1_eval.py` · `phase2.py` · `phase3.py` · `requirements.txt` · `README.md`
```

***