# 🐕 Pipeline YOLO + ViT untuk Deteksi dan Klasifikasi Ras Anjing

## Overview

Pipeline 2 tahap untuk mendeteksi dan mengklasifikasikan ras anjing:
1. **YOLO** - Mendeteksi lokasi anjing dalam gambar (bounding box)
2. **ViT** - Mengklasifikasikan ras anjing dari gambar yang sudah di-crop

## Struktur File Baru

```
DogBreedViT/
├── crop_dataset_with_bbox.py    # Crop dataset menggunakan XML annotation
├── yolo_detector.py             # Setup dan testing YOLO
├── inference_yolo_vit.py        # Pipeline inference YOLO + ViT
├── train_vit_cropped.py         # Training ViT dengan dataset cropped
└── data_cropped/                # Output dataset yang sudah di-crop
    ├── train/
    ├── val/
    └── test/
```

## Langkah-langkah Penggunaan

### Langkah 1: Install Dependencies

```bash
pip install ultralytics  # YOLOv8
```

### Langkah 2: Crop Dataset dengan Bounding Box

Menggunakan file XML annotation yang sudah ada:

```bash
python crop_dataset_with_bbox.py
```

Output:
- Folder `data_cropped/` berisi gambar yang sudah di-crop berdasarkan bounding box
- Struktur: `data_cropped/train/`, `data_cropped/val/`, `data_cropped/test/`

### Langkah 3: Training ViT dengan Dataset Cropped

```bash
python train_vit_cropped.py
```

Output:
- Model: `vit_cropped_best_model.pth`
- Log: `training_cropped_log.csv`
- Visualisasi: `training_curve_cropped.png`, `confusion_matrix_cropped.png`

### Langkah 4: Setup YOLO untuk Inference

```bash
python yolo_detector.py
```

Ini akan:
- Download pretrained YOLOv8
- Test deteksi pada gambar sample

### Langkah 5: Inference pada Gambar Baru

```bash
python inference_yolo_vit.py
```

Atau gunakan dalam kode Python:

```python
from inference_yolo_vit import *

# Load models
yolo_model = load_yolo_model()
vit_model = load_vit_model("vit_cropped_best_model.pth")

# Single image inference
results = inference_pipeline("path/to/image.jpg", yolo_model, vit_model)

# Batch inference
all_results = batch_inference("folder/with/images", yolo_model, vit_model)
```

## Alur Kerja Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALUR KERJA LENGKAP                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TRAINING:                                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                                                         │    │
│  │  1. dataset/ + annotation/                              │    │
│  │         │                                               │    │
│  │         ▼                                               │    │
│  │  2. crop_dataset_with_bbox.py                           │    │
│  │         │                                               │    │
│  │         ▼                                               │    │
│  │  3. data_cropped/                                       │    │
│  │         │                                               │    │
│  │         ▼                                               │    │
│  │  4. train_vit_cropped.py                                │    │
│  │         │                                               │    │
│  │         ▼                                               │    │
│  │  5. vit_cropped_best_model.pth                          │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  INFERENCE:                                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                                                         │    │
│  │  Gambar Baru                                            │    │
│  │         │                                               │    │
│  │         ▼                                               │    │
│  │  YOLO (deteksi) ──→ Bounding Box                        │    │
│  │         │                                               │    │
│  │         ▼                                               │    │
│  │  Crop gambar                                            │    │
│  │         │                                               │    │
│  │         ▼                                               │    │
│  │  ViT (klasifikasi) ──→ Ras Anjing                       │    │
│  │         │                                               │    │
│  │         ▼                                               │    │
│  │  Output: BBox + Breed + Confidence                      │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Output Inference

Untuk setiap anjing yang terdeteksi:
- **Bounding Box**: Koordinat [x1, y1, x2, y2]
- **Detection Confidence**: Tingkat kepercayaan deteksi YOLO
- **Breed**: Ras anjing yang diprediksi
- **Breed Confidence**: Tingkat kepercayaan klasifikasi ViT

## 10 Ras Anjing yang Didukung

1. Bernese Mountain Dog
2. Boxer
3. Doberman
4. German Shepherd
5. Golden Retriever
6. Great Dane
7. Labrador Retriever
8. Rottweiler
9. Samoyed
10. Siberian Husky
