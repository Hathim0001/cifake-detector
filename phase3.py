# =============================================================================
# Phase 3 — Improve: Adversarial Training Defense
# Build → Break → Improve | CIFAKE Dataset
#
# Fix: TensorListDataset returns plain int labels (not torch.tensor)
#      so ConcatDataset collation works with ImageFolder subsets.
# Fix: Cache adversarial training data to avoid regeneration on re-run.
#
# Phase 2 Findings:
#   PGD   → 100% evasion  → primary threat → must defend
#   Blur  → 90%  evasion  → secondary      → augmentation defense
#   FGSM  → 0%   evasion
#   JPEG  → 0%   evasion
#
# Defense: Adversarial Fine-tuning
#   Mix clean + PGD adversarial + blur-augmented FAKE images in training.
#   Fine-tune 3 epochs with low LR to preserve clean accuracy.
# =============================================================================

# %% ── SECTION 1: IMPORTS ─────────────────────────────────────────────────────
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import (
    DataLoader, Dataset, Subset, ConcatDataset
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)


SEED             = 42
DATA_DIR         = r"D:\hackathon\data"
ORIG_MODEL_PATH  = "models/best.pth"
ROBUST_PATH      = "models/robust.pth"
IMG_SIZE         = 64
BATCH_SIZE       = 32
CLASS_NAMES      = ["REAL", "FAKE"]

# Adversarial training config
N_ADV_SAMPLES    = 1000
N_TEST_ADV       = 200
FINETUNE_EPOCHS  = 3
LR_FINETUNE      = 5e-5
PGD_EPS_TRAIN    = 0.03
PGD_ITERS_TRAIN  = 10
BLUR_K_TRAIN     = 11

# Cache paths (avoids regenerating PGD data on re-run)
PGD_IMGS_CACHE   = "models/pgd_train_imgs.pt"
PGD_LBLS_CACHE   = "models/pgd_train_lbls.pt"
BLUR_IMGS_CACHE  = "models/blur_train_imgs.pt"
BLUR_LBLS_CACHE  = "models/blur_train_lbls.pt"
ADV_TEST_CACHE   = "models/adv_test_imgs.pt"
ADV_TEST_L_CACHE = "models/adv_test_lbls.pt"
BLR_TEST_CACHE   = "models/blur_test_imgs.pt"
BLR_TEST_L_CACHE = "models/blur_test_lbls.pt"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

os.makedirs("outputs/phase3", exist_ok=True)

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")

def flip_label(y):
    return 1 - y

eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# %% ── SECTION 4: LOAD DATA ───────────────────────────────────────────────────
full_train_aug  = datasets.ImageFolder(
    TRAIN_DIR, transform=train_transform, target_transform=flip_label
)
full_train_eval = datasets.ImageFolder(
    TRAIN_DIR, transform=eval_transform, target_transform=flip_label
)
test_ds = datasets.ImageFolder(
    TEST_DIR, transform=eval_transform, target_transform=flip_label
)

# Same SEED as Phase 1 → identical split
n_total    = len(full_train_aug)
val_size   = int(n_total * 0.15)
train_size = n_total - val_size
all_idx    = list(range(n_total))
random.shuffle(all_idx)
train_idx  = all_idx[:train_size]
val_idx    = all_idx[train_size:]

train_ds = Subset(full_train_aug,  train_idx)
val_ds   = Subset(full_train_eval, val_idx)

test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, pin_memory=False
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, pin_memory=False
)

print(f"Train: {len(train_ds):,} Val: {len(val_ds):,} Test: {len(test_ds):,}")

# %% ── SECTION 5: MODEL BUILDER ───────────────────────────────────────────────
def build_model():
    model    = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_feats, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2),
    )
    return model.to(DEVICE)

# %% ── SECTION 6: LOAD ORIGINAL MODEL ────────────────────────────────────────
original_model = build_model()
original_model.load_state_dict(torch.load(ORIG_MODEL_PATH, map_location=DEVICE))
original_model.eval()
print("Original model loaded.\n")


def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)

def ensure_3d(t):
    return t.squeeze(0) if (t.dim() == 4 and t.shape[0] == 1) else t

def ensure_4d(t):
    return t.unsqueeze(0) if t.dim() == 3 else t

def get_probs(model, x):
    """Accept [C,H,W] or [1,C,H,W]. Returns (p_real, p_fake)."""
    with torch.no_grad():
        inp   = ensure_4d(x).to(DEVICE)
        probs = F.softmax(model(inp), dim=1)[0]
    return probs[0].item(), probs[1].item()

def tensor_to_uint8(t):
    img = denorm(ensure_3d(t)).permute(1, 2, 0).detach().cpu().numpy()
    return (img * 255).astype(np.uint8)

# %% ── SECTION 8: ATTACK FUNCTIONS ───────────────────────────────────────────
def blur_attack(img_tensor, kernel_size):
    """Gaussian blur. Input [C,H,W] or [1,C,H,W]. Returns [C,H,W]."""
    u8  = tensor_to_uint8(img_tensor)
    k   = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    bl  = cv2.GaussianBlur(u8, (k, k), sigmaX=0).astype(np.float32) / 255.0
    bl  = (bl - 0.5) / 0.5
    return torch.tensor(bl).permute(2, 0, 1).float()

def pgd_attack(model, img_tensor, epsilon, alpha=0.004,
               n_iters=40, target_class=0):
    """Targeted PGD. Returns ([C,H,W], conf_history)."""
    model.eval()
    x     = ensure_4d(img_tensor).clone().detach().to(DEVICE)
    label = torch.tensor([target_class]).to(DEVICE)
    x_adv = (x + torch.empty_like(x).uniform_(
        -epsilon * 0.5, epsilon * 0.5)).clamp(-1.0, 1.0).detach()
    history = []
    for _ in range(n_iters):
        x_adv = x_adv.requires_grad_(True)
        loss  = F.cross_entropy(model(x_adv), label)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv - alpha * x_adv.grad.sign()
            delta = (x_adv - x).clamp(-epsilon, epsilon)
            x_adv = (x + delta).clamp(-1.0, 1.0)
        _, pfake = get_probs(model, x_adv)
        history.append(pfake)
    return ensure_3d(x_adv).cpu(), history


class TensorListDataset(Dataset):
    """
    Wraps lists of image tensors and labels.
    Returns plain int labels — consistent with ImageFolder
    so ConcatDataset collation works correctly.
    """
    def __init__(self, imgs, labels):
        self.imgs   = imgs
        self.labels = labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Return plain int label (NOT torch.tensor) to match ImageFolder
        return self.imgs[idx], int(self.labels[idx])

# %% ── SECTION 10: EVALUATE HELPERS ──────────────────────────────────────────
def evaluate_model(model, loader, desc="Evaluating"):
    """Full metrics on any DataLoader."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    p, l = np.array(all_preds), np.array(all_labels)
    return {
        "acc":    accuracy_score(l, p),
        "prec":   precision_score(l, p, pos_label=1, zero_division=0),
        "rec":    recall_score(l, p, pos_label=1, zero_division=0),
        "f1":     f1_score(l, p, pos_label=1, zero_division=0),
        "cm":     confusion_matrix(l, p),
        "preds":  p,
        "labels": l,
    }

def eval_on_tensor_loader(model, loader, desc=""):
    """Accuracy + F1 on a TensorListDataset loader."""
    model.eval()
    all_preds, all_lbls = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs = imgs.to(DEVICE)
            if isinstance(labels, torch.Tensor):
                labels = labels.to(DEVICE)
            else:
                labels = torch.tensor(labels).to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_lbls.extend(labels.cpu().numpy())
    p, l = np.array(all_preds), np.array(all_lbls)
    acc  = accuracy_score(l, p)
    f1   = f1_score(l, p, pos_label=1, zero_division=0)
    return acc, f1, p, l

# %% ── SECTION 11: BASELINE — ORIGINAL MODEL ─────────────────────────────────
print("\n====== Evaluating ORIGINAL model on clean test set")
orig_clean = evaluate_model(original_model, test_loader, "Clean Test")
print(f"  Accuracy : {orig_clean['acc']:.4f}")
print(f"  F1       : {orig_clean['f1']:.4f}\n")

# %% ── SECTION 12: BUILD ADVERSARIAL TEST SET (cache-aware) ──────────────────
if (os.path.exists(ADV_TEST_CACHE) and os.path.exists(BLR_TEST_CACHE)):
    print("Loading cached adversarial test set...")
    adv_test_imgs  = torch.load(ADV_TEST_CACHE)
    adv_test_lbls  = torch.load(ADV_TEST_L_CACHE)
    blur_test_imgs = torch.load(BLR_TEST_CACHE)
    blur_test_lbls = torch.load(BLR_TEST_L_CACHE)
    print(f"  Loaded {len(adv_test_imgs)} PGD + {len(blur_test_imgs)} Blur test examples\n")
else:
    print(f"Building adversarial test set ({N_TEST_ADV} examples)...")
    adv_test_imgs, adv_test_lbls   = [], []
    blur_test_imgs, blur_test_lbls = [], []
    count = 0
    for imgs, labels in tqdm(test_loader, desc="Collecting FAKE test targets"):
        for i in range(len(imgs)):
            if labels[i].item() == 1 and count < N_TEST_ADV:
                img = imgs[i]
                _, pfake = get_probs(original_model, img)
                if pfake >= 0.90:
                    pgd_adv, _ = pgd_attack(
                        original_model, img, epsilon=0.05, n_iters=40
                    )
                    adv_test_imgs.append(pgd_adv)
                    adv_test_lbls.append(1)

                    blur_adv = blur_attack(img, 15)
                    blur_test_imgs.append(blur_adv)
                    blur_test_lbls.append(1)
                    count += 1
        if count >= N_TEST_ADV:
            break

    torch.save(adv_test_imgs,  ADV_TEST_CACHE)
    torch.save(adv_test_lbls,  ADV_TEST_L_CACHE)
    torch.save(blur_test_imgs, BLR_TEST_CACHE)
    torch.save(blur_test_lbls, BLR_TEST_L_CACHE)
    print(f"  Built & cached {len(adv_test_imgs)} PGD + {len(blur_test_imgs)} Blur test examples\n")

adv_test_ds  = TensorListDataset(adv_test_imgs,  adv_test_lbls)
blur_test_ds = TensorListDataset(blur_test_imgs, blur_test_lbls)

adv_test_loader = DataLoader(
    adv_test_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, pin_memory=False
)
blur_test_loader = DataLoader(
    blur_test_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, pin_memory=False
)

# Original model baseline on adversarial test
print("Evaluating ORIGINAL model on adversarial test set...")
orig_pgd_acc,  orig_pgd_f1,  _, _ = eval_on_tensor_loader(
    original_model, adv_test_loader, "PGD Test (Original)"
)
orig_blur_acc, orig_blur_f1, _, _ = eval_on_tensor_loader(
    original_model, blur_test_loader, "Blur Test (Original)"
)
print(f"  PGD  adversarial accuracy : {orig_pgd_acc:.4f}  F1: {orig_pgd_f1:.4f}")
print(f"  Blur adversarial accuracy : {orig_blur_acc:.4f}  F1: {orig_blur_f1:.4f}\n")

# %% ── SECTION 13: GENERATE ADVERSARIAL TRAINING DATA (cache-aware) ──────────
if (os.path.exists(PGD_IMGS_CACHE) and os.path.exists(BLUR_IMGS_CACHE)):
    print("Loading cached adversarial training data...")
    pgd_train_imgs  = torch.load(PGD_IMGS_CACHE)
    blur_train_imgs = torch.load(BLUR_IMGS_CACHE)
    pgd_train_lbls  = torch.load(PGD_LBLS_CACHE)
    blur_train_lbls = torch.load(BLUR_LBLS_CACHE)
    print(f"  Loaded {len(pgd_train_imgs)} PGD + {len(blur_train_imgs)} Blur train examples\n")
else:
    print(f"Generating {N_ADV_SAMPLES} adversarial training examples...")

    # Collect FAKE images from training set
    fake_train_imgs = []
    for img, label in tqdm(
        Subset(full_train_eval, train_idx), desc="Collecting FAKE train images"
    ):
        if label == 1:
            fake_train_imgs.append(img)
        if len(fake_train_imgs) >= N_ADV_SAMPLES:
            break
    print(f"  Collected {len(fake_train_imgs)} FAKE training images")

    # PGD adversarial examples (still labeled FAKE)
    pgd_train_imgs, pgd_train_lbls   = [], []
    blur_train_imgs, blur_train_lbls = [], []

    print("  Generating PGD adversarial training examples...")
    for img in tqdm(fake_train_imgs, desc="PGD (train)"):
        adv, _ = pgd_attack(
            original_model, img,
            epsilon=PGD_EPS_TRAIN,
            alpha=0.004,
            n_iters=PGD_ITERS_TRAIN,
            target_class=0
        )
        pgd_train_imgs.append(adv)
        pgd_train_lbls.append(1)     # still FAKE

    print("  Generating Blur augmentation training examples...")
    for img in tqdm(fake_train_imgs, desc="Blur (train)"):
        bl = blur_attack(img, BLUR_K_TRAIN)
        blur_train_imgs.append(bl)
        blur_train_lbls.append(1)    # still FAKE

    # Save cache
    torch.save(pgd_train_imgs,  PGD_IMGS_CACHE)
    torch.save(pgd_train_lbls,  PGD_LBLS_CACHE)
    torch.save(blur_train_imgs, BLUR_IMGS_CACHE)
    torch.save(blur_train_lbls, BLUR_LBLS_CACHE)
    print(f"\n  Saved adversarial training cache to models/")

print(f"  PGD  adversarial training samples : {len(pgd_train_imgs):,}")
print(f"  Blur adversarial training samples : {len(blur_train_imgs):,}")

# %% ── SECTION 14: BUILD COMBINED TRAINING SET ────────────────────────────────
pgd_train_ds  = TensorListDataset(pgd_train_imgs,  pgd_train_lbls)
blur_train_ds = TensorListDataset(blur_train_imgs, blur_train_lbls)

# Subset of original training (manageable on CPU)
original_subset_size = min(5000, len(train_ds))
orig_subset_idx      = random.sample(range(len(train_ds)), original_subset_size)
orig_subset_ds       = Subset(train_ds, orig_subset_idx)

# ConcatDataset: all return plain int labels — collation works correctly
combined_ds = ConcatDataset([orig_subset_ds, pgd_train_ds, blur_train_ds])

combined_loader = DataLoader(
    combined_ds, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0, pin_memory=False
)

print(f"\nCombined Training Set:")
print(f"  Original (subset) : {len(orig_subset_ds):,}")
print(f"  PGD adversarial   : {len(pgd_train_ds):,}")
print(f"  Blur augmented    : {len(blur_train_ds):,}")
print(f"  Total             : {len(combined_ds):,}")

# %% ── SECTION 15: ADVERSARIAL FINE-TUNING ────────────────────────────────────
robust_model = build_model()
robust_model.load_state_dict(torch.load(ORIG_MODEL_PATH, map_location=DEVICE))

for p in robust_model.parameters():
    p.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(robust_model.parameters(), lr=LR_FINETUNE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

best_f1      = 0.0
best_weights = None
history      = {
    "train_loss": [], "val_loss": [],
    "train_acc":  [], "val_acc":  []
}

print(f"\n====== Adversarial Fine-tuning ({FINETUNE_EPOCHS} epochs)")

for epoch in range(1, FINETUNE_EPOCHS + 1):

    # ── Train ─────────────────────────────────────────────────────────────────
    robust_model.train()
    t_loss, t_correct, t_total = 0.0, 0, 0

    for imgs, labels in tqdm(
        combined_loader, desc=f"Epoch {epoch}/{FINETUNE_EPOCHS} Train", leave=False
    ):
        imgs = imgs.to(DEVICE)
        # Handle both int labels (from ImageFolder) and tensor labels
        if isinstance(labels, torch.Tensor):
            labels = labels.to(DEVICE)
        else:
            labels = torch.tensor(labels, dtype=torch.long).to(DEVICE)

        logits = robust_model(imgs)
        loss   = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds      = logits.argmax(1)
        t_loss    += loss.item() * len(labels)
        t_correct += (preds == labels).sum().item()
        t_total   += len(labels)

    t_loss /= t_total
    t_acc   = t_correct / t_total

    # ── Validate ──────────────────────────────────────────────────────────────
    robust_model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    v_preds_all, v_lbls_all    = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(
            val_loader, desc=f"Epoch {epoch}/{FINETUNE_EPOCHS} Val  ", leave=False
        ):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = robust_model(imgs)
            loss   = criterion(logits, labels)
            preds  = logits.argmax(1)
            v_loss    += loss.item() * len(labels)
            v_correct += (preds == labels).sum().item()
            v_total   += len(labels)
            v_preds_all.extend(preds.cpu().numpy())
            v_lbls_all.extend(labels.cpu().numpy())

    v_loss /= v_total
    v_acc   = v_correct / v_total
    v_f1    = f1_score(v_lbls_all, v_preds_all, pos_label=1, zero_division=0)
    scheduler.step()

    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)
    history["train_acc"].append(t_acc)
    history["val_acc"].append(v_acc)

    print(
        f"  Epoch {epoch}/{FINETUNE_EPOCHS}"
        f" | Train Loss: {t_loss:.4f}  Acc: {t_acc:.4f}"
        f" |  Val  Loss: {v_loss:.4f}  Acc: {v_acc:.4f}  F1: {v_f1:.4f}"
    )

    if v_f1 > best_f1:
        best_f1      = v_f1
        best_weights = deepcopy(robust_model.state_dict())
        torch.save(best_weights, ROBUST_PATH)
        print(f"    [SAVED] Best robust model — Val F1 = {best_f1:.4f}")

robust_model.load_state_dict(best_weights)
robust_model.eval()
print(f"\nRobust model saved → {ROBUST_PATH}")

# %% ── SECTION 16: FINAL EVALUATION ──────────────────────────────────────────
print("\n====== Evaluating ROBUST model")

print("  Clean test set...")
rob_clean = evaluate_model(robust_model, test_loader, "Clean Test (Robust)")
print(f"    Accuracy : {rob_clean['acc']:.4f}   F1: {rob_clean['f1']:.4f}")

print("  PGD adversarial test set...")
rob_pgd_acc,  rob_pgd_f1,  _, _ = eval_on_tensor_loader(
    robust_model, adv_test_loader, "PGD Test (Robust)"
)
print(f"    Accuracy : {rob_pgd_acc:.4f}   F1: {rob_pgd_f1:.4f}")

print("  Blur adversarial test set...")
rob_blur_acc, rob_blur_f1, _, _ = eval_on_tensor_loader(
    robust_model, blur_test_loader, "Blur Test (Robust)"
)
print(f"    Accuracy : {rob_blur_acc:.4f}   F1: {rob_blur_f1:.4f}")

# %% ── SECTION 17: TRAINING CURVES ───────────────────────────────────────────
def plot_training_curves(history,
                         save_path="outputs/phase3/finetune_curves.png"):
    eps  = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(eps, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(eps, history["val_loss"],   "r-s", label="Val Loss")
    axes[0].set_title("Fine-tuning Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps, history["train_acc"], "b-o", label="Train Acc")
    axes[1].plot(eps, history["val_acc"],   "r-s", label="Val Acc")
    axes[1].set_title("Fine-tuning Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Phase 3: Adversarial Fine-tuning Curves",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")

plot_training_curves(history)

# %% ── SECTION 18: ROBUSTNESS COMPARISON CHART ───────────────────────────────
def plot_comparison(save_path="outputs/phase3/robustness_comparison.png"):
    categories = ["Clean\nTest Set", "PGD\nAdversarial", "Blur\nAdversarial"]
    orig_accs  = [orig_clean["acc"], orig_pgd_acc,  orig_blur_acc]
    rob_accs   = [rob_clean["acc"],  rob_pgd_acc,   rob_blur_acc]
    x     = np.arange(len(categories))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bars1 = axes[0].bar(x - width/2, [v * 100 for v in orig_accs],
                         width, label="Original Model", color="steelblue",   alpha=0.85)
    bars2 = axes[0].bar(x + width/2, [v * 100 for v in rob_accs],
                         width, label="Robust Model",   color="darkorange",  alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=11)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 115)
    axes[0].set_title("Accuracy: Original vs Robust Model",
                       fontweight="bold", fontsize=12)
    axes[0].axhline(50, color="red", linestyle="--",
                    linewidth=1, label="Random baseline")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2, axis="y")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1,
                     f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    deltas = [(r - o) * 100 for o, r in zip(orig_accs, rob_accs)]
    colors = ["green" if d >= 0 else "tomato" for d in deltas]
    bars3  = axes[1].bar(categories, deltas, color=colors,
                          alpha=0.85, edgecolor="black")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Accuracy Improvement (%)")
    axes[1].set_title("Improvement: Robust − Original\n(positive = defense works)",
                       fontweight="bold", fontsize=12)
    axes[1].grid(True, alpha=0.2, axis="y")

    for bar, val in zip(bars3, deltas):
        ypos = val + 0.5 if val >= 0 else val - 2.5
        axes[1].text(bar.get_x() + bar.get_width() / 2, ypos,
                     f"{val:+.1f}%", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")

    plt.suptitle("Phase 3: Defense Effectiveness",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")

plot_comparison()

# %% ── SECTION 19: CONFUSION MATRICES ────────────────────────────────────────
def plot_confusion_matrices(
        save_path="outputs/phase3/confusion_matrices.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, result, title in zip(
        axes,
        [orig_clean, rob_clean],
        ["Original Model", "Robust Model"]
    ):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=result["cm"], display_labels=CLASS_NAMES
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(
            f"{title}\nAcc={result['acc']:.4f}  F1={result['f1']:.4f}",
            fontsize=11, fontweight="bold"
        )
    plt.suptitle("Confusion Matrices: Clean Test Set",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")

plot_confusion_matrices()

# %% ── SECTION 20: ADVERSARIAL PREDICTIONS COMPARISON ───────────────────────
def plot_adversarial_predictions(
        save_path="outputs/phase3/adversarial_predictions.png"):
    """
    Show 5 PGD adversarial examples side-by-side.
    Left: Original model prediction | Right: Robust model prediction.
    Prefer examples where robust model fixes the original's mistake.
    """
    original_model.eval()
    robust_model.eval()

    examples = []
    # First try: find cases where robust model fixed the error
    for i in range(len(adv_test_imgs)):
        img_adv = adv_test_imgs[i]
        _, orig_pf = get_probs(original_model, img_adv)
        _, rob_pf  = get_probs(robust_model,   img_adv)
        if orig_pf < 0.5 and rob_pf >= 0.5:
            examples.append({
                "adv": img_adv, "orig_pf": orig_pf, "rob_pf": rob_pf,
                "orig_pred": "REAL", "rob_pred": "FAKE",
            })
        if len(examples) >= 5:
            break

    # Fallback: show first 5 regardless
    if len(examples) == 0:
        print("  Showing first 5 adversarial examples (fallback).")
        for i in range(min(5, len(adv_test_imgs))):
            img_adv = adv_test_imgs[i]
            _, orig_pf = get_probs(original_model, img_adv)
            _, rob_pf  = get_probs(robust_model,   img_adv)
            examples.append({
                "adv": img_adv, "orig_pf": orig_pf, "rob_pf": rob_pf,
                "orig_pred": "REAL" if orig_pf < 0.5 else "FAKE",
                "rob_pred":  "REAL" if rob_pf  < 0.5 else "FAKE",
            })

    n   = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(7, n * 3.2))
    axes      = axes.reshape(n, 2)

    axes[0][0].set_title("Original Model", fontsize=11, fontweight="bold")
    axes[0][1].set_title("Robust Model",   fontsize=11, fontweight="bold")

    for row, ex in enumerate(examples):
        disp = denorm(ensure_3d(ex["adv"])).permute(1, 2, 0).numpy()

        axes[row][0].imshow(disp)
        orig_color = "green" if ex["orig_pred"] == "FAKE" else "red"
        axes[row][0].set_xlabel(
            f"{ex['orig_pred']}  P(fake)={ex['orig_pf']:.3f}",
            fontsize=9, color=orig_color
        )
        axes[row][0].axis("off")

        axes[row][1].imshow(disp)
        rob_color = "green" if ex["rob_pred"] == "FAKE" else "red"
        axes[row][1].set_xlabel(
            f"{ex['rob_pred']}  P(fake)={ex['rob_pf']:.3f}",
            fontsize=9, color=rob_color
        )
        axes[row][1].axis("off")
        axes[row][0].set_ylabel(f"Sample {row+1}", fontsize=9,
                                 rotation=0, labelpad=55, va="center")

    plt.suptitle(
        "PGD Adversarial Examples: Original vs Robust Model\n"
        "Green = correct (FAKE detected)  |  Red = wrong (evaded)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")

plot_adversarial_predictions()

# %% ── SECTION 21: FINAL SUMMARY ─────────────────────────────────────────────
print("""
====== PHASE 3: DEFENSE PROPOSAL & RESULTS =================================

VULNERABILITY (from Phase 2):
  The original model relied on HIGH-FREQUENCY texture artifacts to detect
  synthetic images — not semantic content. This made it fragile:
    PGD  → 100% evasion rate (gradient attack erases HF features directly)
    Blur → 90%  evasion rate (smoothing removes HF artifact fingerprints)

DEFENSE IMPLEMENTED: Adversarial Fine-tuning
  1. Generate 1000 PGD-perturbed FAKE images (labeled FAKE) → model must
     classify them correctly despite gradient-based perturbation.
  2. Generate 1000 Blur-augmented FAKE images (labeled FAKE) → model must
     detect FAKE even after HF artifacts are removed.
  3. Mix with 5000 original clean training samples.
  4. Fine-tune for 3 epochs at low LR (5e-5) to preserve clean accuracy
     while gaining adversarial robustness.

WHY THIS WORKS:
  Adversarial training forces the model to find deeper, more stable features
  beyond just HF noise patterns. When it sees PGD-perturbed FAKE images that
  look almost real, it must learn lower-frequency semantic differences
  (color distributions, spatial coherence, object statistics).
  This is the standard approach used in real-world robust classifiers.
""")

print("RESULTS SUMMARY")
print(f"Condition Orig Acc Robust Acc Improvement")
print("-" * 50)
for name, o, r in [
    ("Clean Test Set",   orig_clean["acc"], rob_clean["acc"]),
    ("PGD Adversarial",  orig_pgd_acc,      rob_pgd_acc),
    ("Blur Adversarial", orig_blur_acc,     rob_blur_acc),
]:
    d    = r - o
    sign = "+" if d >= 0 else ""
    print(f"  {name:<23} {o*100:>9.2f}%  {r*100:>9.2f}%  {sign}{d*100:>10.2f}%")

print("\n====== OUTPUT FILES")
for f in sorted(os.listdir("outputs/phase3")):
    print(f"  outputs/phase3/{f}")

print("\n====== PHASE 3 COMPLETE")
