import os
# Workaround for OpenMP duplicate runtime error on some Windows Conda/PyTorch setups.
# This tells the Intel/OpenMP runtime it's OK to continue when multiple copies are detected.
# Unsafe but pragmatic; if you prefer, set this in your environment instead of in code.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from utils.dataset_loader import get_dataloaders
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

def imshow(img_tensor):
    img = img_tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  # unnormalize agar terlihat normal
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')

def main():
    # 1. Muat DataLoader
    train_loader, val_loader, test_loader = get_dataloaders(data_dir="data", batch_size=8)

    # 2. Ambil satu batch data
    images, labels = next(iter(train_loader))
    print(f"Ukuran batch: {images.shape}")  # contoh output: torch.Size([8, 3, 224, 224])

    # 3. Tampilkan beberapa gambar hasil preprocessing
    plt.figure(figsize=(10, 5))
    for i in range(8):
        plt.subplot(2, 4, i+1)
        imshow(images[i])
        plt.title(f"Label: {labels[i].item()}")
    plt.tight_layout()

    # Save the image instead of showing it in terminal
    output_path = "dataloader_test_output.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Image saved to: {output_path}")
    print("  Open this file to see the preprocessed images.")
    plt.close()

if __name__ == '__main__':
    # On Windows, multiprocessing in DataLoader requires freeze_support
    multiprocessing.freeze_support()
    main()