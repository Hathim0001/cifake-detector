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
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import torchvision.models as models
import torchvision.datasets as datasets

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)
SEED        = 42
DATA_DIR    = r"D:\hackathon\data"
BATCH_SIZE  = 32
IMG_SIZE    = 64
VAL_SPLIT   = 0.15
SAVE_PATH   = "models/best.pth"
CLASS_NAMES = ["REAL", "FAKE"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device {DEVICE}")
print(f"Loading model {SAVE_PATH}")

os.makedirs("outputs", exist_ok=True)

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")

def flip_label(y):
    return 1 - y

eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

test_ds = datasets.ImageFolder(
    TEST_DIR, transform=eval_transform, target_transform=flip_label
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=False
)
print(f"Test set     : {len(test_ds):,} images")

def build_model():
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_feats, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2),
    )
    return model.to(DEVICE)

model = build_model()
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully from checkpoint.\n")

def denorm(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def collect_samples(loader, label_filter=None, n=6):
    imgs_out, lbls_out = [], []
    for imgs, labels in loader:
        for i in range(len(imgs)):
            if label_filter is None or labels[i].item() == label_filter:
                imgs_out.append(imgs[i])
                lbls_out.append(labels[i].item())
            if len(imgs_out) >= n:
                return imgs_out, lbls_out
    return imgs_out, lbls_out

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Evaluating"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

acc  = accuracy_score (all_labels, all_preds)
prec = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
rec  = recall_score   (all_labels, all_preds, pos_label=1, zero_division=0)
f1   = f1_score       (all_labels, all_preds, pos_label=1, zero_division=0)
cm   = confusion_matrix(all_labels, all_preds)

print("Test Set Metrics")
print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Full Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix Test Set", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.show()
print("Saved: outputs/confusion_matrix.png")

class GradCAM:
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

        weights = self.gradients[0].mean(dim=(1, 2))
        cam     = (weights[:, None, None] * self.activations[0]).sum(0)
        cam     = F.relu(cam).cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam     = cv2.resize(cam, (x.shape[-1], x.shape[-2]))
        return cam, class_idx, probs


grad_cam = GradCAM(model, model.layer4[-1])


def visualize_gradcam(loader, n=6, label_filter=None, save_tag="gradcam"):
    c_imgs, c_lbls = collect_samples(loader, label_filter=label_filter, n=n)
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3.2))
    for col, title in enumerate(["Input Image", "Grad-CAM Heatmap", "Overlay"]):
        axes[0][col].set_title(title, fontsize=11, fontweight="bold")

    for i in range(len(c_imgs)):
        img_tensor        = c_imgs[i].unsqueeze(0)
        cam, pred_idx, probs = grad_cam(img_tensor)
        img_disp  = denorm(c_imgs[i]).permute(1, 2, 0).numpy()
        img_uint8 = (img_disp * 255).astype(np.uint8)
        heatmap   = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap   = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap   = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
        overlay   = cv2.addWeighted(img_uint8, 0.55, heatmap, 0.45, 0)
        row_lbl   = (f"True: {CLASS_NAMES[c_lbls[i]]} "
                     f"Pred: {CLASS_NAMES[pred_idx]} "
                     f"P(fake)={probs[1]:.3f}")
        axes[i][0].imshow(img_disp);         axes[i][0].axis("off")
        axes[i][1].imshow(heatmap / 255.0);  axes[i][1].axis("off")
        axes[i][2].imshow(overlay / 255.0);  axes[i][2].axis("off")
        axes[i][0].set_ylabel(row_lbl, fontsize=8, rotation=0,
                               labelpad=140, va="center")

    plt.suptitle("Grad-CAM Visualizations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"outputs/{save_tag}.png", dpi=150)
    plt.show()
    print(f"Saved: outputs/{save_tag}.png")

print("Generating Grad-CAM for FAKE images...")
visualize_gradcam(test_loader, n=6, label_filter=1, save_tag="gradcam_fake")
print("Generating Grad-CAM for REAL images...")
visualize_gradcam(test_loader, n=6, label_filter=0, save_tag="gradcam_real")

def compute_saliency(model, img_tensor):
    model.eval()
    x = img_tensor.to(DEVICE).requires_grad_(True)
    logits     = model(x)
    pred_class = logits.argmax(dim=1).item()
    logits[0, pred_class].backward()
    saliency   = x.grad.data.abs().max(dim=1)[0].squeeze().cpu().numpy()
    saliency   = (saliency - saliency.min()) / (saliency.max() + 1e-8)
    return saliency, pred_class


def visualize_saliency(loader, n=6, label_filter=None, save_tag="saliency"):
    c_imgs, c_lbls = collect_samples(loader, label_filter=label_filter, n=n)
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3.2))
    for col, title in enumerate(["Input Image", "Saliency Map", "Blended Overlay"]):
        axes[0][col].set_title(title, fontsize=11, fontweight="bold")

    for i in range(len(c_imgs)):
        img_tensor        = c_imgs[i].unsqueeze(0)
        saliency, pred_idx = compute_saliency(model, img_tensor)
        img_disp  = denorm(c_imgs[i]).permute(1, 2, 0).numpy()
        img_uint8 = (img_disp * 255).astype(np.uint8)
        sal_color = cv2.applyColorMap((saliency * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        sal_color = cv2.cvtColor(sal_color, cv2.COLOR_BGR2RGB)
        sal_color = cv2.resize(sal_color, (img_uint8.shape[1], img_uint8.shape[0]))
        blended   = cv2.addWeighted(img_uint8, 0.6, sal_color, 0.4, 0)
        row_lbl   = (f"True: {CLASS_NAMES[c_lbls[i]]} "
                     f"Pred: {CLASS_NAMES[pred_idx]}")
        axes[i][0].imshow(img_disp);              axes[i][0].axis("off")
        axes[i][1].imshow(saliency, cmap="hot");  axes[i][1].axis("off")
        axes[i][2].imshow(blended / 255.0);       axes[i][2].axis("off")
        axes[i][0].set_ylabel(row_lbl, fontsize=8, rotation=0,
                               labelpad=140, va="center")

    plt.suptitle("Saliency Maps Pixel-Level Sensitivity",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"outputs/{save_tag}.png", dpi=150)
    plt.show()
    print(f"Saved: outputs/{save_tag}.png")

print("Generating Saliency Maps for FAKE images...")
visualize_saliency(test_loader, n=6, label_filter=1, save_tag="saliency_fake")
print("Generating Saliency Maps for REAL images...")
visualize_saliency(test_loader, n=6, label_filter=0, save_tag="saliency_real")

print("PHASE 1 COMPLETE")
print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Checkpoint : {SAVE_PATH}")
print("Output files:")
for fname in sorted(os.listdir("outputs")):
    print(f"outputs/{fname}")
