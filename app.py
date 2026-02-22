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

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
from PIL import Image
import io
import base64
from pathlib import Path
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

# Thresholds:
# - YOLO_CONF_THRESHOLD (40%): Minimum confidence untuk object detection (< 40% = No dogs detected)
# - BREED_CONF_THRESHOLD (85%): Minimum confidence untuk valid breed (< 85% = Unknown breed)

app = FastAPI(title="Dog Breed Classification")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load models at startup
print("Loading models...")
yolo_model = load_yolo_model()
vit_model = load_vit_model()
print("Models loaded!")


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save temporarily
        temp_path = "temp_upload.jpg"
        image.save(temp_path)
        
        # Run detection
        detections = detect_dogs_yolo(yolo_model, temp_path, conf_thresh=YOLO_CONF_THRESHOLD)
        
        if not detections:
            os.remove(temp_path)
            return JSONResponse({
                "success": False,
                "message": "No dogs detected in the image",
                "detections": []
            })
        
        # Classify each detection
        results = []
        
        for i, det in enumerate(detections):
            cropped = crop_image(image, det['bbox'])
            cls = classify_dog_breed(vit_model, cropped, min_confidence=BREED_CONF_THRESHOLD)
            
            result = {
                'id': i + 1,
                'bbox': det['bbox'],
                'breed': cls['breed'],
                'confidence': round(cls['confidence'] * 100, 2),
                'is_valid': cls['is_valid'],
                'all_probs': {k: round(v * 100, 2) for k, v in cls['all_probs'].items()}
            }
            results.append(result)
        
        # Create visualization
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        colors = plt.cm.Set1(np.linspace(0, 1, 10))
        
        for r in results:
            x1, y1, x2, y2 = r['bbox']
            breed = r['breed']
            conf = r['confidence']
            
            if r['is_valid']:
                color = colors[CLASS_NAMES.index(breed) % 10]
                label = f"{breed}\n{conf:.1f}%"
            else:
                color = (0.5, 0.5, 0.5, 1.0)
                label = "Unknown"
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3,
                                      edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            ax.text(x1, y1-10, label, color='white', fontsize=12, fontweight='bold',
                    va='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8))
        
        ax.axis('off')
        
        # Save visualization
        result_path = "static/result.jpg"
        plt.savefig(result_path, bbox_inches='tight', dpi=120)
        plt.close()
        
        # Clean up
        os.remove(temp_path)
        
        # Count valid detections
        valid_count = sum(1 for r in results if r['is_valid'])
        
        if valid_count > 0:
            message = f"Detected {len(results)} dog(s), {valid_count} identified"
        else:
            message = f"Detected {len(results)} dog(s), but breed not recognized (may not be in trained classes)"
        
        return JSONResponse({
            "success": True,
            "message": message,
            "detections": results,
            "result_image": "/static/result.jpg"
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error: {str(e)}",
            "detections": []
        })


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) #ketik di google "localhost:8000" untuk akses webnya
