# =============================================================================
# Phase 1 Build & Evaluate: Synthetic Image Detector
# CIFAKE Dataset ResNet-18 Transfer Learning
#
import os
import random
from copy import deepcopy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import torchvision.models as models
import torchvision.datasets as datasets

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEVICE}")

DATA_DIR    = r"D:\hackathon\data"
BATCH_SIZE  = 32
EPOCHS_S1   = 5
EPOCHS_S2   = 5
LR_HEAD     = 1e-3
LR_BACKBONE = 1e-4
VAL_SPLIT   = 0.15
IMG_SIZE    = 64
PATIENCE    = 3
SAVE_PATH   = "models/best.pth"

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ImageFolder sorts alphabetically: FAKE=0, REAL=1
# We flip → FAKE=1 (positive/synthetic), REAL=0 (negative/real)
CLASS_NAMES = ["REAL", "FAKE"]

# ── Named function instead of lambda (Windows pickle fix) ────────────────────
def flip_label(y):
    return 1 - y

# %% ──────────────────────────────────────────────────────────────────────────
# SECTION 3: PATH VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")

for path in [DATA_DIR, TRAIN_DIR, TEST_DIR,
             os.path.join(TRAIN_DIR, "REAL"),
             os.path.join(TRAIN_DIR, "FAKE"),
             os.path.join(TEST_DIR,  "REAL"),
             os.path.join(TEST_DIR,  "FAKE")]:
    status = "OK" if os.path.exists(path) else "MISSING"
    print(f"[{status}] {path}")

for folder in [os.path.join(TRAIN_DIR, "REAL"), os.path.join(TRAIN_DIR, "FAKE"),
               os.path.join(TEST_DIR,  "REAL"), os.path.join(TEST_DIR,  "FAKE")]:
    if os.path.exists(folder):
        count = len([f for f in os.listdir(folder)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        label = os.path.basename(folder)
        split = os.path.basename(os.path.dirname(folder))
        print(f"{split}/{label}: {count:,} images")

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

full_train_aug  = datasets.ImageFolder(
    TRAIN_DIR, transform=train_transform, target_transform=flip_label
)
full_train_eval = datasets.ImageFolder(
    TRAIN_DIR, transform=eval_transform, target_transform=flip_label
)
test_ds = datasets.ImageFolder(
    TEST_DIR, transform=eval_transform, target_transform=flip_label
)

# Train / Val split (consistent indices)
n_total    = len(full_train_aug)
val_size   = int(n_total * VAL_SPLIT)
train_size = n_total - val_size

all_indices = list(range(n_total))
random.shuffle(all_indices)
train_idx = all_indices[:train_size]
val_idx   = all_indices[train_size:]

train_ds = Subset(full_train_aug,  train_idx)   # with augmentation
val_ds   = Subset(full_train_eval, val_idx)     # without augmentation

# ── DataLoaders (num_workers=0 for Windows, pin_memory=False for CPU) ────────
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=False
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=False
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=False
)

print(f"Dataset Splits:")
print(f"Train : {len(train_ds):,}")
print(f"Val   : {len(val_ds):,}")
print(f"Test  : {len(test_ds):,}")

def denorm(tensor):
    """Reverse normalization [-1,1] → [0,1] for display."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def show_samples(loader, n=8, title="Dataset Samples"):
    imgs, labels = next(iter(loader))
    imgs, labels = imgs[:n], labels[:n]

    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 2.8))
    for ax, img, lbl in zip(axes, imgs, labels):
        img_disp = denorm(img).permute(1, 2, 0).numpy()
        ax.imshow(img_disp)
        ax.set_title(
            CLASS_NAMES[lbl.item()], fontsize=10,
            color="red" if lbl.item() == 1 else "green"
        )
        ax.axis("off")

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = f"outputs/{title.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


show_samples(train_loader, title="Training Samples")
show_samples(test_loader,  title="Test Samples")


def build_model(freeze_backbone=True):
    """
    ResNet-18 pretrained on ImageNet.
    Stage 1: freeze backbone → train head only.
    Stage 2: unfreeze → fine-tune everything.
    Head: Dropout(0.3) → Linear(512,256) → ReLU → Dropout(0.2) → Linear(256,2)
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_feats = model.fc.in_features   # 512 for ResNet-18
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_feats, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2),             # 2 classes: REAL=0, FAKE=1
    )
    return model.to(DEVICE)


model = build_model(freeze_backbone=True)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Model: ResNet-18")
print(f"Trainable params : {trainable:,}")
print(f"Total params     : {total:,}")
print(f"Head: {model.fc}")

criterion = nn.CrossEntropyLoss()


def make_optimizer(model, stage=1):
    """
    Stage 1 → only head params at LR_HEAD.
    Stage 2 → backbone at LR_BACKBONE, head at LR_HEAD.
    """
    if stage == 1:
        return optim.Adam(model.fc.parameters(), lr=LR_HEAD)
    else:
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
        return optim.Adam([
            {"params": backbone_params,       "lr": LR_BACKBONE},
            {"params": model.fc.parameters(), "lr": LR_HEAD},
        ])


def run_epoch(model, loader, optimizer=None, train=True):
    """
    Single pass over the loader.
    Returns: avg_loss, accuracy, predictions list, labels list
    """
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, labels in tqdm(
            loader, desc="  Train" if train else "  Eval ", leave=False
        ):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds       = logits.argmax(dim=1)
            total_loss += loss.item() * len(labels)
            correct    += (preds == labels).sum().item()
            total      += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def train_model(model, train_loader, val_loader, epochs, stage=1):
    """
    Full training loop with early stopping.
    Best checkpoint saved to SAVE_PATH based on validation F1.
    """
    optimizer = make_optimizer(model, stage=stage)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_f1      = 0.0
    best_weights = None
    patience_ctr = 0
    history      = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  []
    }

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, _, _              = run_epoch(model, train_loader, optimizer, train=True)
        vl_loss, vl_acc, vl_preds, vl_lbl = run_epoch(model, val_loader,              train=False)

        vl_f1 = f1_score(
            vl_lbl, vl_preds, pos_label=1,
            average="binary", zero_division=0
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(
            f"  Epoch {epoch:02d}/{epochs}"
            f"  | Train  Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}"
            f"  |  Val   Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}  F1: {vl_f1:.4f}"
        )

        if vl_f1 > best_f1:
            best_f1      = vl_f1
            best_weights = deepcopy(model.state_dict())
            torch.save(best_weights, SAVE_PATH)
            patience_ctr = 0
            print(f"    [SAVED] Best model — Val F1 = {best_f1:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"    [STOP]  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_weights)
    return model, history

# %% ──────────────────────────────────────────────────────────────────────────
# SECTION 8: TRAIN — Stage 1 then Stage 2
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nStage 1: Classifier Head Only <=====")
model, hist_s1 = train_model(
    model, train_loader, val_loader, epochs=EPOCHS_S1, stage=1
)

print(f"\nStage 2: Fine-tune Full Network <=====")
for param in model.parameters():
    param.requires_grad = True          # unfreeze backbone

model, hist_s2 = train_model(
    model, train_loader, val_loader, epochs=EPOCHS_S2, stage=2
)


def plot_history(h, title="Training History"):
    eps  = range(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(eps, h["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(eps, h["val_loss"],   marker="s", label="Val Loss")
    axes[0].set_title("Loss");     axes[0].set_xlabel("Epoch")
    axes[0].legend();              axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps, h["train_acc"], marker="o", label="Train Acc")
    axes[1].plot(eps, h["val_acc"],   marker="s", label="Val Acc")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend();              axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"outputs/{title.lower().replace(' ', '_').replace(':', '')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


plot_history(hist_s1, "Stage 1: Head Only")
plot_history(hist_s2, "Stage 2: Fine-tuning")

# %% ──────────────────────────────────────────────────────────────────────────
# SECTION 10: EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

model.eval()
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testing"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits  = model(imgs)
        probs   = F.softmax(logits, dim=1)
        preds   = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())   # P(FAKE)

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# ── Compute Metrics ──────────────────────────────────────────────────────────
acc  = accuracy_score (all_labels, all_preds)
prec = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
rec  = recall_score   (all_labels, all_preds, pos_label=1, zero_division=0)
f1   = f1_score       (all_labels, all_preds, pos_label=1, zero_division=0)
cm   = confusion_matrix(all_labels, all_preds)

print(f"\nTest Set Metrics")
print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nFull Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# ── Confusion Matrix ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix Test Set", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.show()
print("Saved: outputs/confusion_matrix.png")

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Hooks into the target layer to capture:
      - activations (forward hook)
      - gradients   (backward hook)
    """
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        x = x.to(DEVICE)

        logits = self.model(x)
        probs  = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pooling of gradients → per-channel weights
        weights = self.gradients[0].mean(dim=(1, 2))                     # [C]
        cam     = (weights[:, None, None] * self.activations[0]).sum(0)  # [H,W]
        cam     = F.relu(cam).cpu().numpy()

        # Normalize to [0,1] and resize to input image dimensions
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]))
        return cam, class_idx, probs


grad_cam = GradCAM(model, model.layer4[-1])


def collect_samples(loader, label_filter=None, n=6):
    """Collect n images (optionally filtered by label) from a loader."""
    imgs_out, lbls_out = [], []
    for imgs, labels in loader:
        for i in range(len(imgs)):
            if label_filter is None or labels[i].item() == label_filter:
                imgs_out.append(imgs[i])
                lbls_out.append(labels[i].item())
            if len(imgs_out) >= n:
                return imgs_out, lbls_out
    return imgs_out, lbls_out


def visualize_gradcam(loader, n=6, label_filter=None, save_tag="gradcam"):
    """
    Visualize Grad-CAM for n images.
    Columns: Input | Grad-CAM Heatmap | Overlay
    """
    c_imgs, c_lbls = collect_samples(loader, label_filter=label_filter, n=n)

    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3.2))
    for col, title in enumerate(["Input Image", "Grad-CAM Heatmap", "Overlay"]):
        axes[0][col].set_title(title, fontsize=11, fontweight="bold")

    for i in range(len(c_imgs)):
        img_tensor = c_imgs[i].unsqueeze(0)
        true_lbl   = c_lbls[i]
        cam, pred_idx, probs = grad_cam(img_tensor)

        img_disp  = denorm(c_imgs[i]).permute(1, 2, 0).numpy()
        img_uint8 = (img_disp * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))

        overlay = cv2.addWeighted(img_uint8, 0.55, heatmap, 0.45, 0)

        row_lbl = (
            f"True: {CLASS_NAMES[true_lbl]} | "
            f"Pred: {CLASS_NAMES[pred_idx]} | "
            f"P(fake)={probs[1]:.3f}"
        )

        axes[i][0].imshow(img_disp);          axes[i][0].axis("off")
        axes[i][1].imshow(heatmap / 255.0);   axes[i][1].axis("off")
        axes[i][2].imshow(overlay / 255.0);   axes[i][2].axis("off")
        axes[i][0].set_ylabel(row_lbl, fontsize=8, rotation=0,
                               labelpad=140, va="center")

    plt.suptitle("Grad-CAM Visualizations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"outputs/{save_tag}.png", dpi=150)
    plt.show()
    print(f"Saved: outputs/{save_tag}.png")


visualize_gradcam(test_loader, n=6, label_filter=1, save_tag="gradcam_fake")
visualize_gradcam(test_loader, n=6, label_filter=0, save_tag="gradcam_real")


def compute_saliency(model, img_tensor):
    """
    Vanilla gradient saliency.
    Backprop from predicted class score to input pixels.
    Returns: saliency [H,W] normalized to [0,1], predicted class index.
    """
    model.eval()
    x = img_tensor.to(DEVICE).requires_grad_(True)

    logits     = model(x)
    pred_class = logits.argmax(dim=1).item()
    logits[0, pred_class].backward()

    # Max over RGB channels, take absolute value
    saliency = x.grad.data.abs().max(dim=1)[0].squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() + 1e-8)
    return saliency, pred_class


def visualize_saliency(loader, n=6, label_filter=None, save_tag="saliency"):
    """
    Visualize saliency maps for n images.
    Columns: Input | Saliency (hot) | Blended Overlay
    """
    c_imgs, c_lbls = collect_samples(loader, label_filter=label_filter, n=n)

    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3.2))
    for col, title in enumerate(["Input Image", "Saliency Map", "Blended Overlay"]):
        axes[0][col].set_title(title, fontsize=11, fontweight="bold")

    for i in range(len(c_imgs)):
        img_tensor = c_imgs[i].unsqueeze(0)
        true_lbl   = c_lbls[i]
        saliency, pred_idx = compute_saliency(model, img_tensor)

        img_disp  = denorm(c_imgs[i]).permute(1, 2, 0).numpy()
        img_uint8 = (img_disp * 255).astype(np.uint8)

        sal_color = cv2.applyColorMap(
            (saliency * 255).astype(np.uint8), cv2.COLORMAP_HOT
        )
        sal_color = cv2.cvtColor(sal_color, cv2.COLOR_BGR2RGB)
        sal_color = cv2.resize(sal_color, (img_uint8.shape[1], img_uint8.shape[0]))

        blended = cv2.addWeighted(img_uint8, 0.6, sal_color, 0.4, 0)

        row_lbl = (
            f"True: {CLASS_NAMES[true_lbl]} | "
            f"Pred: {CLASS_NAMES[pred_idx]}"
        )

        axes[i][0].imshow(img_disp);             axes[i][0].axis("off")
        axes[i][1].imshow(saliency, cmap="hot"); axes[i][1].axis("off")
        axes[i][2].imshow(blended / 255.0);      axes[i][2].axis("off")
        axes[i][0].set_ylabel(row_lbl, fontsize=8, rotation=0,
                               labelpad=140, va="center")

    plt.suptitle("Saliency Maps — Pixel-Level Sensitivity",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"outputs/{save_tag}.png", dpi=150)
    plt.show()
    print(f"Saved: outputs/{save_tag}.png")


visualize_saliency(test_loader, n=6, label_filter=1, save_tag="saliency_fake")
visualize_saliency(test_loader, n=6, label_filter=0, save_tag="saliency_real")

print("PHASE 1 COMPLETE")
print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Model saved to {SAVE_PATH}")
print("Output files:")
for f in sorted(os.listdir("outputs")):
    print(f"outputs/{f}")
