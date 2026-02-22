# Deploy ke Render.com

Tutorial lengkap untuk deploy Dog Breed Classification ke Render.com

## 🚀 Quick Start (TL;DR)

```bash
# 1. Upload model ke Hugging Face
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload YOUR_USERNAME/dog-breed-vit vit_merged_best_model.pth --repo-type=model

# 2. Update app.py dengan URL Hugging Face (lihat app_render_example.py)

# 3. Push ke GitHub
git init
git add .
git commit -m "Deploy to Render"
git push

# 4. Deploy ke Render.com
# - Connect GitHub repo
# - Build: pip install -r requirements_web.txt
# - Start: uvicorn app:app --host 0.0.0.0 --port $PORT
```

## Persiapan

### 1. Ukuran Model Files
Model Anda terlalu besar untuk GitHub:
- `vit_merged_best_model.pth` (~330MB)
- `yolov8n.pt` (~6MB)

**Solusi: Upload model ke cloud storage**

### 2. Upload Model ke Hugging Face Hub (Recommended)

**Hugging Face adalah platform terbaik untuk hosting ML models:**
- ✅ Free hosting unlimited
- ✅ Direct download reliable
- ✅ Khusus untuk ML models
- ✅ Version control built-in
- ✅ Fast CDN (Content Delivery Network)

#### Step-by-Step Upload ke Hugging Face:

**A. Install Hugging Face CLI**
```bash
pip install huggingface_hub
```

**B. Login ke Hugging Face**
```bash
# Buat akun gratis di https://huggingface.co/join
# Dapatkan token di https://huggingface.co/settings/tokens

huggingface-cli login
# Paste your token when prompted
```

**C. Upload Model**
```bash
# Pastikan berada di folder yang ada model files
cd C:\DogBreedViT

# Upload ViT model
huggingface-cli upload bry11/dog-breed-vit vit_merged_best_model.pth --repo-type=model

# Upload YOLO model (optional, karena sudah ada di GitHub)
huggingface-cli upload bry11/dog-breed-vit yolov8n.pt --repo-type=model
```

**Ganti `YOUR_USERNAME` dengan username Hugging Face Anda!**

**D. Get Direct Download URL**

Setelah upload berhasil, URL direct download:
```
https://huggingface.co/YOUR_USERNAME/dog-breed-vit/resolve/main/vit_merged_best_model.pth
https://huggingface.co/YOUR_USERNAME/dog-breed-vit/resolve/main/yolov8n.pt
```

**Contoh:**
```python
VIT_MODEL_URL = "https://huggingface.co/johndoe/dog-breed-vit/resolve/main/vit_merged_best_model.pth"
YOLO_MODEL_URL = "https://huggingface.co/johndoe/dog-breed-vit/resolve/main/yolov8n.pt"
```

#### Alternative: Upload via Web Interface

1. Buka https://huggingface.co/new (buat model baru)
2. Nama repo: `dog-breed-vit`
3. Klik **Files** → **Add file** → Upload `vit_merged_best_model.pth`
4. Wait for upload to complete
5. Copy URL dari file yang sudah diupload

---

### Alternative: Google Drive / Dropbox (Jika tidak pakai HuggingFace)

<details>
<summary>Klik untuk lihat cara Google Drive/Dropbox</summary>

#### Google Drive (⚠️ Tidak recommended untuk file >100MB)

1. Upload kedua file model ke Google Drive
2. Set sharing ke "Anyone with the link can view"
3. Dapatkan direct download links:

**Format link Google Drive:**
```
https://drive.google.com/file/d/FILE_ID/view?usp=sharing
```

**Ubah menjadi direct download:**
```
https://drive.google.com/uc?export=download&id=FILE_ID
```

**⚠️ Warning:** Google Drive sering block direct download untuk file besar!

#### Dropbox (Better alternative)

```
# Link sharing biasa:
https://www.dropbox.com/s/abc123xyz/vit_model.pth?dl=0

# Ubah menjadi direct download (ganti dl=0 ke dl=1):
https://www.dropbox.com/s/abc123xyz/vit_model.pth?dl=1
```

#### GitHub Releases (max 2GB per file)

Bisa digunakan untuk host model di GitHub Releases.

</details>

## Setup Repository GitHub

### 1. Buat `.gitignore`

Buat file `.gitignore` di root folder:
```
# Model files (terlalu besar untuk GitHub)
*.pth
*.pt

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so

# Virtual environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data folders
data/
data_merged/
dataset/
crawled_images/
annotation/
Google/
example/
history_tuning/

# Logs and temp files
*.log
*.csv
temp_upload.jpg
static/result.jpg
training_*.png
confusion_matrix*.png

# Jupyter
.ipynb_checkpoints/
*.ipynb
```

### 2. Modifikasi `app.py` untuk Download Model

Tambahkan fungsi download model di awal `app.py` (sebelum import inference_yolo_vit):

```python
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

# GANTI dengan URL Hugging Face Anda!
VIT_MODEL_URL = "https://huggingface.co/YOUR_USERNAME/dog-breed-vit/resolve/main/vit_merged_best_model.pth"
YOLO_MODEL_URL = "https://huggingface.co/YOUR_USERNAME/dog-breed-vit/resolve/main/yolov8n.pt"

# Atau gunakan YOLO dari official repo:
# YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

print("Checking models...")
download_model(VIT_MODEL_URL, "vit_merged_best_model.pth")
download_model(YOLO_MODEL_URL, "yolov8n.pt")
print("All models ready!")

# Lanjutkan dengan import dan code lainnya...
# from fastapi import FastAPI, File, UploadFile
# ...
```

**Catatan:** Letakkan kode ini SEBELUM `from inference_yolo_vit import ...` agar model sudah tersedia saat import.

### 3. Push ke GitHub

```bash
git init
git add .
git commit -m "Initial commit for Render deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

## Deploy ke Render.com

### 1. Sign Up & Create New Web Service

1. Buka [render.com](https://render.com)
2. Sign up (bisa pakai GitHub account)
3. Klik **"New +"** → **"Web Service"**
4. Connect GitHub repository Anda

### 2. Konfigurasi Web Service

**Build Settings:**
- **Name:** `dog-breed-classification` (atau nama lain)
- **Environment:** `Python 3`
- **Region:** Singapore (terdekat dengan Indonesia)
- **Branch:** `main`
- **Build Command:**
  ```bash
  pip install -r requirements_web.txt
  ```
- **Start Command:**
  ```bash
  uvicorn app:app --host 0.0.0.0 --port $PORT
  ```

**Instance Settings:**
- **Instance Type:** Free (750 hours/month)
  - ⚠️ RAM terbatas (512MB), model loading mungkin lambat
  - Untuk performa lebih baik, gunakan Starter ($7/month) dengan 2GB RAM

### 3. Environment Variables (Optional)

Jika ingin URL model di environment variable agar mudah diubah:

**Di app.py, ganti hardcoded URL dengan:**
```python
import os

VIT_MODEL_URL = os.getenv(
    "VIT_MODEL_URL", 
    "https://huggingface.co/YOUR_USERNAME/dog-breed-vit/resolve/main/vit_merged_best_model.pth"
)
YOLO_MODEL_URL = os.getenv(
    "YOLO_MODEL_URL",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
)
```

**Di Render Dashboard, tambahkan environment variables:**
- Key: `VIT_MODEL_URL`
- Value: `https://huggingface.co/YOUR_USERNAME/dog-breed-vit/resolve/main/vit_merged_best_model.pth`

- Key: `YOLO_MODEL_URL`
- Value: `https://huggingface.co/YOUR_USERNAME/dog-breed-vit/resolve/main/yolov8n.pt`

**Keuntungan:** Bisa update URL tanpa commit code lagi.

### 4. Deploy

1. Klik **"Create Web Service"**
2. Tunggu build process (5-10 menit pertama kali)
3. Setelah selesai, akan dapat URL: `https://your-app-name.onrender.com`

## Testing

1. Buka URL Render: `https://your-app-name.onrender.com`
2. Upload gambar anjing untuk test
3. Pastikan deteksi dan klasifikasi berjalan dengan baik

## Troubleshooting

### Model Download Gagal

**Masalah:** Google Drive block direct download untuk file besar

**Solusi:**
1. Gunakan GitHub Releases untuk host model
2. Atau gunakan Dropbox dengan parameter `?raw=1`
3. Atau split model files dan join saat startup

### Out of Memory (OOM)

**Masalah:** Free tier RAM terbatas (512MB)

**Solusi:**
1. Upgrade ke Starter plan ($7/month, 2GB RAM)
2. Atau optimize model (quantization, pruning)
3. Atau gunakan model size lebih kecil

### Cold Start Lambat

**Masalah:** Free tier sleep setelah 15 menit tidak aktif

**Solusi:**
1. First request setelah sleep butuh waktu lebih lama
2. Keep-alive script (ping setiap 10 menit) - ⚠️ Melawan TOS Render
3. Upgrade ke paid plan (tidak sleep)

### Build Timeout

**Masalah:** Build melebihi 15 menit (free tier)

**Solusi:**
1. Pre-download dependencies di local, test dulu
2. Atau gunakan Docker image yang sudah pre-built

## Alternatif Hosting

Jika Render tidak cocok:

### Railway.app
- Pros: $5 credit gratis, lebih generous resource
- Cons: Setelah credit habis, bayar per usage

### Fly.io
- Pros: Free tier lumayan, performa bagus
- Cons: Setup lebih teknis (perlu Dockerfile)

### Hugging Face Spaces
- Pros: Free GPU, cocok untuk ML apps
- Cons: Perlu refactor ke Gradio/Streamlit interface

### DigitalOcean App Platform
- Pros: Professional, reliable
- Cons: Berbayar (~$5/month minimum)

## Tips Production

1. **Add Health Check Endpoint** (sudah ada di `app.py`):
   ```python
   @app.get("/health")
   async def health():
       return {"status": "healthy"}
   ```

2. **Enable CORS** (sudah ada):
   - Sudah configured untuk accept all origins

3. **Add Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

4. **Monitor Usage**:
   - Render dashboard menampilkan CPU, Memory, Request stats
   - Set up alerts jika perlu

5. **Custom Domain** (Optional):
   - Bisa connect custom domain di Render settings
   - Contoh: `dogbreed.yourdomain.com`

## Biaya Estimasi

### Free Tier (Render.com)
- 750 jam/bulan gratis
- 512MB RAM
- Shared CPU
- Sleep after 15 min inactivity
- **Cost:** $0/month

### Starter Tier (Recommended)
- 2GB RAM
- Tidak sleep
- Better performance
- **Cost:** $7/month

## Kesimpulan

Untuk **tugas akhir/demo**, gunakan:
- ✅ **Free tier Render** - cukup untuk showcase
- ✅ URL permanen untuk dokumentasi

Untuk **production/real usage**:
- ✅ **Paid tier** - performa lebih baik
- ✅ **VPS** jika butuh full control

Selamat mencoba! 🚀
