# Contoh Modifikasi app.py untuk Render Deployment dengan Hugging Face
# Tambahkan kode ini di bagian atas app.py sebelum import inference_yolo_vit

import os
import requests
from pathlib import Path

def download_model(url, filename):
    """Download model jika belum ada"""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f:
                if total_size == 0:
                    # Jika tidak ada content-length, download langsung
                    f.write(response.content)
                    print(f"{filename} downloaded!")
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = (downloaded / total_size) * 100
                            print(f"  Progress: {progress:.1f}%", end='\r')
                    print(f"\n{filename} downloaded!")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            if os.path.exists(filename):
                os.remove(filename)
            raise
    else:
        print(f"{filename} already exists")

# URL untuk download models dari Hugging Face
# GANTI YOUR_USERNAME dengan username Hugging Face Anda
# Contoh: https://huggingface.co/johndoe/dog-breed-vit/resolve/main/vit_merged_best_model.pth

VIT_MODEL_URL = os.getenv(
    "VIT_MODEL_URL", 
    "https://huggingface.co/bry11/dog-breed-vit/blob/main/vit_merged_best_model.pth"
)

YOLO_MODEL_URL = os.getenv(
    "YOLO_MODEL_URL",
    "https://huggingface.co/bry11/dog-breed-vit/blob/main/yolov8n.pt"
)

# Download models at startup (sebelum import inference_yolo_vit)
print("=" * 60)
print("Checking and downloading models...")
print("=" * 60)

try:
    download_model(VIT_MODEL_URL, "vit_merged_best_model.pth")
    download_model(YOLO_MODEL_URL, "yolov8n.pt")
    print("All models ready!")
except Exception as e:
    print(f"Failed to download models: {e}")
    print("Please check your model URLs")
    print("Make sure you've uploaded models to Hugging Face and updated the URLs")

# ==============================================================================
# Setelah ini, lanjutkan dengan kode app.py yang sudah ada
# ==============================================================================

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Import functions from inference_yolo_vit
from inference_yolo_vit import (
    load_yolo_model, load_vit_model, detect_dogs_yolo,
    crop_image, classify_dog_breed, CLASS_NAMES,
    YOLO_CONF_THRESHOLD, BREED_CONF_THRESHOLD
)

# ... sisanya sama seperti app.py original Anda

