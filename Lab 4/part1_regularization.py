"""
ESE 577 - Lab 4, Part 1: Dropout Regularization with LeNet
Dataset: CIFAR-10
Experiments: dropout p in {0, 0.3, 0.5, 0.7} x data fraction in {1.0, 0.5, 0.25}
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hyperparameters (kept constant across all experiments) ───────────────────
LR          = 1e-3
BATCH_SIZE  = 128
NUM_EPOCHS  = 30
NUM_CLASSES = 10   # CIFAR-10

# Experiment axes
DROPOUT_PROBS  = [0.0, 0.3, 0.5, 0.7]
DATA_FRACTIONS = [1.0, 0.5, 0.25]


# ── LeNet with Dropout ────────────────────────────────────────────────────────
class LeNetDropout(nn.Module):
    """
    LeNet-5 adapted for 32x32 RGB images (CIFAR-10).
    Dropout is applied only in the fully-connected layers.
    """
    def __init__(self, num_classes: int = 10, dropout_p: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 6, kernel_size=5),   # 32→28
            nn.ReLU(),
            nn.AvgPool2d(2, 2),               # 28→14
            # Block 2
            nn.Conv2d(6, 16, kernel_size=5),  # 14→10
            nn.ReLU(),
            nn.AvgPool2d(2, 2),               # 10→5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Data helpers ──────────────────────────────────────────────────────────────
def get_dataloaders(data_fraction: float):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    val_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_val)

    # Subset training data
    n_total = len(full_train)
    n_use   = int(n_total * data_fraction)
    indices = random.sample(range(n_total), n_use)
    train_subset = Subset(full_train, indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set,      batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader


# ── Training / evaluation ─────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    """One epoch of train (optimizer given) or eval (optimizer=None)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


def train_model(dropout_p: float, data_fraction: float):
    print(f"\n  dropout={dropout_p}, data={int(data_fraction*100)}%", flush=True)
    train_loader, val_loader = get_dataloaders(data_fraction)

    model     = LeNetDropout(NUM_CLASSES, dropout_p).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    for epoch in range(NUM_EPOCHS):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = run_epoch(model, val_loader,   criterion)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        if (epoch + 1) % 5 == 0:
            print(f"    ep {epoch+1:2d} | "
                  f"TL={t_loss:.4f} TA={t_acc:.3f} | "
                  f"VL={v_loss:.4f} VA={v_acc:.3f}")
    return history


if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    # ── Run all experiments ───────────────────────────────────────────────────
    print("=" * 60)
    print("Part 1: LeNet Dropout Regularization Study")
    print("=" * 60)

    all_results = {}   # {(dropout_p, data_fraction): history}
    for frac in DATA_FRACTIONS:
        for dp in DROPOUT_PROBS:
            key = (dp, frac)
            all_results[key] = train_model(dp, frac)

    # ── Plotting ──────────────────────────────────────────────────────────────
    EPOCHS = list(range(1, NUM_EPOCHS + 1))
    os.makedirs("figures", exist_ok=True)

    for frac in DATA_FRACTIONS:
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        fig.suptitle(f"LeNet Dropout Study — {int(frac*100)}% Training Data",
                     fontsize=14, fontweight="bold")

        for col, dp in enumerate(DROPOUT_PROBS):
            h = all_results[(dp, frac)]

            # Loss
            ax = axes[0, col]
            ax.plot(EPOCHS, h["train_loss"], label="Train")
            ax.plot(EPOCHS, h["val_loss"],   label="Val",  linestyle="--")
            ax.set_title(f"p={dp}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss" if col == 0 else "")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            # Accuracy
            ax = axes[1, col]
            ax.plot(EPOCHS, h["train_acc"], label="Train")
            ax.plot(EPOCHS, h["val_acc"],   label="Val",  linestyle="--")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy" if col == 0 else "")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        fname = f"figures/lenet_dropout_data{int(frac*100)}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {fname}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Data%':>6} {'Dropout':>8} {'Best Val Acc':>14} {'Final Gap (T-V)':>16}")
    print("-" * 70)
    for frac in DATA_FRACTIONS:
        for dp in DROPOUT_PROBS:
            h = all_results[(dp, frac)]
            best_val = max(h["val_acc"])
            gap      = h["train_acc"][-1] - h["val_acc"][-1]   # overfitting gap
            print(f"{int(frac*100):>5}% {dp:>8.1f} {best_val:>14.4f} {gap:>+16.4f}")
    print("=" * 70)

    # ── Analysis: effect of dropout vs data fraction ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Best Validation Accuracy: Dropout × Data Fraction",
                 fontsize=13, fontweight="bold")
    for i, frac in enumerate(DATA_FRACTIONS):
        best_vals = [max(all_results[(dp, frac)]["val_acc"]) for dp in DROPOUT_PROBS]
        axes[i].bar([str(p) for p in DROPOUT_PROBS], best_vals, color="#4C72B0")
        axes[i].set_title(f"{int(frac*100)}% Training Data")
        axes[i].set_xlabel("Dropout Probability")
        axes[i].set_ylabel("Best Val Accuracy")
        axes[i].set_ylim(0, 1)
        axes[i].grid(axis="y", alpha=0.3)
        for j, v in enumerate(best_vals):
            axes[i].text(j, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    fname = "figures/lenet_dropout_summary.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")