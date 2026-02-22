# Dog Breed Classification Web Application

Web application untuk deteksi dan klasifikasi ras anjing menggunakan YOLOv8 dan Vision Transformer (ViT).

## 📋 Daftar Isi

- [Cara Menjalankan Lokal](#cara-menjalankan)
- [Deploy ke Internet](#deploy-ke-internet)
- [Fitur](#fitur)
- [Supported Dog Breeds](#supported-dog-breeds)
- [Troubleshooting](#troubleshooting)

## Deploy ke Internet

**Ingin teman-teman akses dari mana saja tanpa jaringan yang sama?**

📘 **[Tutorial Deploy ke Render.com](DEPLOY_RENDER.md)** - Hosting permanen di cloud (gratis)
📘 **[Tutorial Ngrok](DEPLOY_NGROK.md)** - Quick demo dalam 5 menit (paling mudah!)

Tutorial lengkap untuk deploy aplikasi ini ke internet, pilih sesuai kebutuhan:
- **Ngrok**: Paling cepat untuk demo/testing (< 5 menit setup)
- **Render**: URL permanen untuk showcase/production

## Cara Menjalankan

### 1. Persiapan

**Install Dependencies:**
```bash
pip install -r requirements_web.txt
```

**Pastikan model sudah ada:**
- `vit_merged_best_model.pth` (model ViT)
- `yolov8n.pt` (model YOLO)

### 2. Jalankan Server

```bash
python app.py
```

**Output yang muncul:**
```
Loading models...
Loading YOLO: yolov8n.pt
Loading ViT: vit_merged_best_model.pth
Models loaded!
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 3. Akses dari Desktop/Laptop

Buka browser dan akses: **http://localhost:8000**

### 4. Akses dari Mobile (HP/Tablet)

Untuk membuka website dari HP/tablet yang terhubung ke jaringan yang sama:

**A. Cek IP Address Komputer:**

**Windows:**
```bash
ipconfig
```
Cari baris **IPv4 Address**, contoh: `192.168.1.100`

**Mac/Linux:**
```bash
ifconfig | grep "inet "
# atau
ip addr show
```

**B. Pastikan HP dan Komputer di Jaringan yang Sama:**
- HP dan komputer harus terhubung ke **WiFi yang sama**
- Atau gunakan hotspot HP, lalu komputer connect ke hotspot tersebut

**C. Buka Browser di HP:**

Ketik di address bar: `http://[IP-KOMPUTER]:8000`

Contoh: `http://192.168.1.100:8000`

**D. Troubleshooting jika tidak bisa akses:**

1. **Matikan Windows Firewall** (sementara):
   - Buka Windows Defender Firewall
   - Pilih "Turn off Windows Defender Firewall" untuk Private network
   
2. **Atau buat rule firewall:**
   ```bash
   netsh advfirewall firewall add rule name="Python Server" dir=in action=allow protocol=TCP localport=8000
   ```

3. **Cek server berjalan di 0.0.0.0 (bukan 127.0.0.1)**:
   - File `app.py` sudah dikonfigurasi dengan `host="0.0.0.0"`
   - Ini penting agar bisa diakses dari device lain

## Struktur File

```
DogBreedViT/
├── app.py                      # FastAPI backend
├── inference_yolo_vit.py       # Model inference logic
├── vit_merged_best_model.pth   # Model ViT
├── yolov8n.pt                  # Model YOLO
├── static/
│   ├── index.html             # Frontend UI
│   ├── style.css              # Styling
│   ├── script.js              # JavaScript logic
│   └── result.jpg             # Output visualization
└── models/
    └── vit_model.py           # ViT architecture
```

## Fitur

### Input
- **Upload gambar** dari file (drag & drop atau click)
- **Capture foto** langsung dari kamera (mobile dan desktop)
- Support format: JPG, PNG, JPEG

### Deteksi & Klasifikasi
- **Deteksi anjing otomatis** dengan YOLOv8
- **Klasifikasi ras** dengan Vision Transformer
- **Multi-dog detection** - bisa detect beberapa anjing dalam 1 gambar

### Output
- **Visualisasi** dengan bounding box
- **Confidence score** setiap deteksi (hanya untuk breed yang teridentifikasi)
- **Prediksi ras tertinggi** untuk anjing yang teridentifikasi (>= 85% confidence)
- **Warning Unknown** untuk anjing dengan ras tidak dikenali (< 85% confidence, tanpa confidence score)
- **No dogs detected** untuk gambar tanpa anjing atau bukan anjing

### UI/UX
- **Responsive design** - otomatis menyesuaikan layar mobile/tablet/desktop
- **Drag & drop** upload yang mudah
- **Camera modal** dengan video preview untuk capture foto
- **Loading indicator** saat processing
- **Error handling** yang informatif

## Threshold Detection

Sistem menggunakan 2-tier threshold:

### 1. YOLO Detection (40%)
- **< 40% confidence**: "No dogs detected"
- **>= 40% confidence**: Anjing terdeteksi, lanjut ke klasifikasi breed

### 2. Breed Classification (85%)
- **< 85% confidence**: "Unknown" breed (anjing terdeteksi tapi ras tidak dikenali)
  - Menampilkan "Unknown" tanpa confidence score
  - Menampilkan warning: breed not in trained classes
- **>= 85% confidence**: Breed identified
  - Menampilkan nama ras dengan confidence score

**Catatan:**
- YOLO threshold 40% dipilih agar bisa mendeteksi anjing kecil (Chihuahua, dll)
- Breed threshold 85% untuk memastikan prediksi yang sangat akurat dan mengurangi false positive
- Jika gambar bukan anjing sama sekali, YOLO akan memberikan confidence < 40%

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML + CSS + JavaScript
- **Model**: YOLOv8 + Vision Transformer
- **Preprocessing**: Pillow, Torchvision
- **Visualization**: Matplotlib

## Supported Dog Breeds

Model dilatih untuk mengenali **10 ras anjing**:

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

Ras lain akan dideteksi sebagai "Unknown" (jika masih anjing) atau "No dogs detected" (jika bukan anjing).

## Deploy ke Internet

### Opsi Deployment

**Ingin aplikasi diakses dari mana saja via internet?**

#### 1. Render.com (Recommended)
- ✅ Free tier 750 jam/bulan
- ✅ URL permanen
- ✅ Tutorial lengkap: **[DEPLOY_RENDER.md](DEPLOY_RENDER.md)**

#### 2. Ngrok (Quick Demo)
- ✅ Paling cepat (< 5 menit)
- ✅ Gratis untuk testing
- ❌ URL berubah setiap restart
- Tutorial:
  ```bash
  # Jalankan server
  python app.py
  
  # Di terminal lain
  ngrok http 8000
  ```
  Share URL yang muncul: `https://xxxx.ngrok-free.app`

#### 3. Railway.app
- ✅ $5 credit gratis
- ✅ Setup mudah
- Mirip dengan Render

#### 4. VPS (DigitalOcean, AWS, dll)
- ✅ Full control
- ❌ Berbayar (~$5-10/month)
- Untuk production/profesional

**Lihat tutorial lengkap di [DEPLOY_RENDER.md](DEPLOY_RENDER.md)**

### Deploy untuk Local Network

Sudah dikonfigurasi! Server otomatis berjalan di `0.0.0.0:8000` sehingga bisa diakses dari device lain di jaringan yang sama.

**Cara akses:**
1. Jalankan `python app.py` di komputer
2. Cek IP komputer dengan `ipconfig` (Windows) atau `ifconfig` (Mac/Linux)
3. Di device lain, akses: `http://[IP-KOMPUTER]:8000`

### Catatan Deploy

- **Model size**: ViT model (~330MB) dan YOLO model (~6MB)
- **RAM requirement**: Minimal 2GB untuk inferencing
- **GPU support**: Otomatis detect CUDA jika tersedia (tapi CPU sudah cukup cepat)

## Troubleshooting

### Server tidak bisa diakses dari HP

1. **Cek firewall**: Matikan Windows Firewall atau buat exception untuk port 8000
2. **Cek network**: Pastikan HP dan komputer di WiFi yang sama
3. **Cek IP**: Gunakan IPv4 Address yang benar dari `ipconfig`
4. **Cek server**: Pastikan server berjalan di `0.0.0.0` bukan `127.0.0.1`

### Camera tidak berfungsi

1. **Desktop**: Browser perlu HTTPS atau localhost untuk akses camera
   - Gunakan `localhost:8000` bukan IP address saat akses dari desktop
2. **Mobile**: Camera capture otomatis menggunakan native camera
3. **Permission**: Izinkan browser untuk akses camera

### Gambar anjing tidak terdeteksi

1. **Ukuran anjing kecil**: YOLO threshold sudah 40%, tapi mungkin perlu lebih rendah
2. **Kualitas gambar**: Pastikan gambar tidak blur dan anjing terlihat jelas
3. **Posisi anjing**: Anjing terpotong atau tertutup objek lain mungkin tidak terdeteksi

### Error saat upload gambar

1. **Format**: Pastikan format JPG/PNG/JPEG
2. **Ukuran**: Gambar terlalu besar (>10MB) mungkin lambat, resize dulu
3. **Corrupt file**: Coba gambar lain untuk test
