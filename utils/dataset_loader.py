import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# === DATA LOADER ===
def get_dataloaders(data_dir, batch_size=16):
    """
    Load dataset dengan augmentasi lanjutan.
    """

    # Augmentasi kuat untuk training - BALANCED for better performance
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Less aggressive
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # Moderate rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Balanced jitter
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Subtle translation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Use ImageNet stats
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))  # Moderate erasing
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Use ImageNet stats
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train", 
        transform=train_transforms)
    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val",
        transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test", 
        transform=val_test_transforms)

    # DataLoader with reduced num_workers to prevent Windows issues
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True, 
        drop_last=True)  # drop_last prevents small batches
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True)
    
    print(f"\n DataLoader siap digunakan!")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"Jumlah kelas: {len(train_dataset.classes)} -> {train_dataset.classes}")
    
    return train_loader, val_loader, test_loader
