import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import time
import csv
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import multiprocessing
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns

from utils.dataset_loader import get_dataloaders
from models.vit_model import build_vit_model


# Hasil dari hyperparameter tuning (Trial 5 - Best: 98.02%)
BEST_PARAMS = {
    'lr': 4.458143620091819e-05,
    'weight_decay': 0.018512176610058655,
    'batch_size': 32,
    'label_smoothing': 0.11382827565426205,
    'freeze_ratio': 0.7314443047417426,
    'optimizer': 'AdamW'
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data_merged"
EPOCHS = 50
EARLY_STOP_PATIENCE = 5
BATCH_SIZE = BEST_PARAMS['batch_size']
LR = BEST_PARAMS['lr']
WEIGHT_DECAY = BEST_PARAMS['weight_decay']
NUM_CLASSES = 10
SAVE_PATH = "vit_merged_best_model.pth"
LOG_FILE = "training_merged_log.csv"


def check_dataset():
    """Periksa dataset."""
    if not os.path.exists(DATA_DIR):
        print(" ERROR: Dataset tidak ditemukan!")
        print(f"   Folder '{DATA_DIR}' tidak ada.")
        return False
    
    # Cek struktur folder
    required_folders = ['train', 'val', 'test']
    for folder in required_folders:
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path):
            print(f" ERROR: Folder '{folder}' tidak ditemukan di {DATA_DIR}")
            return False
    
    # Hitung jumlah gambar
    total_images = 0
    for split in required_folders:
        split_path = os.path.join(DATA_DIR, split)
        for class_folder in os.listdir(split_path):
            class_path = os.path.join(split_path, class_folder)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_images += len(images)
    
    if total_images == 0:
        print(" ERROR: Tidak ada gambar di dataset!")
        return False
    
    print(f" Dataset ditemukan: {total_images} gambar")
    return True


def train_epoch(model, loader, optimizer, scheduler, criterion):
    """Training untuk satu epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Forward pass
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Statistik
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    acc = 100 * correct / total
    return total_loss / len(loader), acc


def validate_epoch(model, loader, criterion):
    """Validasi untuk satu epoch."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    acc = 100 * correct / total
    return total_loss / len(loader), acc


def main():
    """Main training function."""
    print("=" * 60)
    print(" TRAINING VIT DENGAN DATASET MERGED")
    print("=" * 60)
    
    # Cek dataset
    if not check_dataset():
        return
    
    print("\n Hyperparameters:")
    for key, value in BEST_PARAMS.items():
        print(f"   {key}: {value}")
    print(f"   epochs: {EPOCHS}")
    print(f"   device: {DEVICE}")
    print(f"   data_dir: {DATA_DIR}")
    print()
    
    # Load dataset
    print(" Loading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE
    )
    
    # Build model
    print("\n Building ViT model...")
    model = build_vit_model(
        num_classes=NUM_CLASSES,
        freeze_ratio=BEST_PARAMS['freeze_ratio']
    )
    model.to(DEVICE)
    print(f" Model loaded on {DEVICE}")
    
    # Optimizer dan scheduler
    optimizer = AdamW(model.parameters(), 
                      lr=LR, 
                      weight_decay=WEIGHT_DECAY)
    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),  # fixed 10% warmup
        num_training_steps=num_training_steps
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=BEST_PARAMS['label_smoothing']
    )
    
    # Backup log lama
    if os.path.exists(LOG_FILE):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_file = f"training_cropped_log_backup_{timestamp}.csv"
        os.rename(LOG_FILE, backup_file)
        print(f" Previous log backed up to: {backup_file}\n")
    
    # Training loop dengan early stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "epoch_time": []}
    
    print(f"\n Mulai training di {DEVICE.upper()} untuk max {EPOCHS} epoch...")
    print(f" Early stopping patience: {EARLY_STOP_PATIENCE} epochs\n")
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        print(f"\n=== EPOCH {epoch+1}/{EPOCHS} ===")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        
        elapsed = time.time() - start_time
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(elapsed)
        
        gap = train_acc - val_acc
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"Gap: {gap:+.2f}% | Time: {elapsed/60:.2f} min | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Log ke CSV
        with open(LOG_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(["Epoch", "Train Loss", "Val Loss", "Train Acc", "Val Acc", "Gap", "Time (s)", "LR"])
            writer.writerow([epoch+1, train_loss, val_loss, train_acc, val_acc, gap, round(elapsed, 2), scheduler.get_last_lr()[0]])
        
        # Simpan model terbaik berdasarkan val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f" Model terbaik disimpan! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")
        else:
            no_improve += 1
            print(f" No improvement ({no_improve}/{EARLY_STOP_PATIENCE})")
        
        # Early stopping
        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n Early stopping! Val loss tidak improve selama {EARLY_STOP_PATIENCE} epochs")
            break
    
    print("\n Training selesai.")
    print(f" Model terbaik tersimpan di: {SAVE_PATH}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_curve_merged.png")
    print(" Training curves saved to: training_curve_merged.png")
    
    # Evaluasi pada test set
    print("\n" + "=" * 60)
    print(" FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
    model.eval()
    
    y_true, y_pred = [], []
    test_correct, test_total = 0, 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1)
            
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    test_acc = 100 * test_correct / test_total
    print(f"\n Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    print("\n Classification Report:")
    print(classification_report(y_true, y_pred, target_names=test_loader.dataset.classes, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_loader.dataset.classes,
                yticklabels=test_loader.dataset.classes)
    plt.title(f'Confusion Matrix - Test Acc: {test_acc:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix_merged.png', dpi=300)
    print(" Saved confusion matrix to: confusion_matrix_merged.png")
    
    # Summary
    print("\n" + "=" * 60)
    print(" TRAINING AND EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\n FINAL TEST ACCURACY: {test_acc:.2f}%")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Total Training Time: {sum(history['epoch_time'])/60:.2f} minutes")
    print(f"\n Output files:")
    print(f"   - Model: {SAVE_PATH}")
    print(f"   - Training log: {LOG_FILE}")
    print(f"   - Training curves: training_curve_merged.png")
    print(f"   - Confusion matrix: confusion_matrix_merged.png")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
