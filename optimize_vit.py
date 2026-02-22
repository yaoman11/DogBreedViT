import os
# Workaround for OpenMP duplicate runtime error on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Suppress TensorFlow info messages
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import optuna
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import multiprocessing
from utils.dataset_loader import get_dataloaders
from models.vit_model import build_vit_model
import numpy as np
import random
import csv
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data_merged"
NUM_CLASSES = 10
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5


# Helper reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def objective(trial):
    # HYPERPARAMETERS
    lr = trial.suggest_float("lr", 1e-5, 8e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.08, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    label_smoothing = trial.suggest_float("label_smoothing", 0.05, 0.15)
    freeze_ratio = trial.suggest_float("freeze_ratio", 0.5, 0.75)
    optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    epochs = MAX_EPOCHS

    print(f"\n{'='*60}")
    print(f" Trial {trial.number}: Testing hyperparameters...")
    print(f"  lr={lr:.2e}, wd={weight_decay:.2e}, bs={batch_size}")
    print(f"  freeze={freeze_ratio:.2f}, label_smoothing={label_smoothing:.2f}")
    print(f"  optimizer={optimizer_type}")
    print(f"{'='*60}")

    # Create CSV log file for this trial
    log_filename = f"optuna_trial_{trial.number}_log.csv"
    with open(log_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Trial', 'Epoch', 'Train_Loss', 'Val_Loss', 'Train_Acc', 'Val_Acc', 'Gap', 'LR', 'Time_s'])

    # DATA LOADER
    print(" Loading dataset...")
    train_loader, val_loader, _ = get_dataloaders(DATA_DIR, batch_size=batch_size)

    # MODEL
    print(" Building ViT model...")
    model = build_vit_model(
        num_classes=NUM_CLASSES,
        freeze_ratio=freeze_ratio
    ).to(DEVICE)
    print(f" Model loaded on {DEVICE}")

    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_training_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),  # fixed 10% warmup
        num_training_steps=num_training_steps,
    )

    # Tambahkan Label Smoothing Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # TRAINING & VALIDATION
    def train_epoch():
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total += labels.size(0)
        return total_loss / len(train_loader), 100 * correct / total

    def validate_epoch():
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return total_loss / len(val_loader), 100 * correct / total

    # TRAINING DENGAN EARLY STOPPING (based on val_loss)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    no_improve = 0

    print(f"\n  Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc = train_epoch()
        val_loss, val_acc = validate_epoch()
        elapsed = time.time() - start_time

        gap = train_acc - val_acc
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print(f"                Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}% | Gap={gap:+.2f}%")
        print(f"                LR={current_lr:.2e}, Time={elapsed:.2f}s")
        
        # Log to CSV
        with open(log_filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([trial.number, epoch+1, train_loss, val_loss, train_acc, val_acc, gap, current_lr, round(elapsed, 2)])

        # Early Stopping based on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1

        trial.report(val_loss, epoch)

        if trial.should_prune() and epoch >= 3:
            print(f"  Trial pruned at epoch {epoch+1}")
            raise optuna.exceptions.TrialPruned()

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f" Early stopping: val_loss tidak improve selama {EARLY_STOP_PATIENCE} epochs")
            break

    print(f" Trial {trial.number} done. Best Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%\n")
    return best_val_acc


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    print(f"\n Starting Optuna hyperparameter optimization on {DEVICE.upper()}")
    print(f" Will run 25 trials with max {MAX_EPOCHS} epochs each")
    print(f" This may take several hours...\n")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25,)  # 25 trials

    print("\n Optimasi selesai!")
    trial = study.best_trial
    print(f"  - Best Validation Accuracy: {trial.value:.2f}%")
    print("  - Best Parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    with open("best_hyperparameters.txt", "w") as f:
        f.write(f"Best Accuracy: {trial.value:.2f}%\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
