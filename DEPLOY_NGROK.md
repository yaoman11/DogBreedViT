# Deploy dengan Ngrok (Quick Demo)

Tutorial lengkap menggunakan Ngrok untuk share aplikasi Dog Breed Classification ke internet dengan cepat.

## Kelebihan Ngrok

✅ **Setup sangat cepat** (< 5 menit)
✅ **Gratis** untuk basic usage
✅ **HTTPS otomatis** (SSL certificate included)
✅ **Cocok untuk demo/testing** dan share ke teman
✅ **Tidak perlu upload ke cloud** - langsung dari komputer Anda

## Kekurangan Ngrok

❌ **URL berubah** setiap restart (gratis plan)
❌ **Koneksi timeout** setelah 2 jam (gratis plan)
❌ **Komputer harus tetap menyala** dan server harus terus berjalan
❌ **Bandwidth terbatas** (gratis: 40 connections/minute)

## Instalasi Ngrok

### 1. Download Ngrok

**Windows:**
1. Kunjungi: https://ngrok.com/download
2. Download versi Windows (ZIP file)
3. Extract ngrok.exe ke folder (contoh: `C:\ngrok\`)

**Atau via Chocolatey:**
```powershell
choco install ngrok
```

**Atau via Scoop:**
```powershell
scoop install ngrok
```

### 2. Sign Up & Get Auth Token

1. Buat akun gratis di: https://dashboard.ngrok.com/signup
2. Setelah login, copy **Auth Token** dari: https://dashboard.ngrok.com/get-started/your-authtoken
3. Authenticate ngrok:
   ```powershell
   ngrok config add-authtoken YOUR_AUTH_TOKEN_HERE
   ```

## Cara Menggunakan

### Metode 1: Basic Usage (Paling Mudah)

#### Step 1: Jalankan Server Aplikasi

Buka **Terminal/PowerShell** pertama:
```powershell
cd C:\DogBreedViT
python app.py
```

Tunggu sampai muncul:
```
Loading models...
Models loaded!
INFO: Uvicorn running on http://0.0.0.0:8000
```

#### Step 2: Jalankan Ngrok

Buka **Terminal/PowerShell** kedua (jangan tutup yang pertama):
```powershell
ngrok http 8000
```

#### Step 3: Get Public URL

Akan muncul tampilan seperti ini:
```
ngrok                                          

Session Status                online
Account                       your@email.com
Version                       3.x.x
Region                        Asia Pacific (ap)
Latency                       35ms
Web Interface                 http://127.0.0.1:4040
Forwarding                    https://xxxx-xx-xx-xx-xx.ngrok-free.app -> http://localhost:8000

Connections                   ttl     opn     rt1     rt5     p50     p90
                              0       0       0.00    0.00    0.00    0.00
```

**URL Publik Anda:** `https://xxxx-xx-xx-xx-xx.ngrok-free.app`

#### Step 4: Share & Test

1. Copy URL `https://xxxx-xx-xx-xx-xx.ngrok-free.app`
2. Share ke teman-teman
3. Mereka bisa langsung akses dari browser
4. Test dengan upload gambar anjing

### Metode 2: Custom Domain (Paid Plan)

Jika punya ngrok paid plan, bisa pakai custom domain:
```powershell
ngrok http --domain=your-custom-domain.ngrok.io 8000
```

### Metode 3: Run in Background

Jika ingin ngrok run in background (Windows):
```powershell
Start-Process ngrok -ArgumentList "http 8000" -WindowStyle Hidden
```

## Tips & Tricks

### 1. Ngrok Web Interface

Buka http://127.0.0.1:4040 untuk:
- ✅ Lihat semua request yang masuk
- ✅ Inspect request/response details
- ✅ Replay requests
- ✅ Monitor traffic

Sangat berguna untuk **debugging**!

### 2. Keep Server Running

Server dan ngrok harus terus berjalan:
- ❌ Jangan close terminal
- ❌ Jangan shutdown/sleep komputer
- ✅ Gunakan laptop dengan charger terpasang
- ✅ Disable auto-sleep di Windows settings

### 3. Restart Ngrok

Jika ngrok timeout atau error, cukup:
```powershell
# Stop ngrok: Ctrl+C
# Start lagi:
ngrok http 8000
```

**Catatan:** URL akan berubah setiap restart!

### 4. Stable URL (Paid Plan)

Untuk URL yang tetap sama:
- Upgrade ke **Ngrok Pro** ($8/month)
- Fitur: Reserved domain, static URL, no timeout

### 5. Port Forwarding Alternative

Jika network Anda block ngrok, coba:
- **LocalTunnel:** `npx localtunnel --port 8000`
- **Serveo:** `ssh -R 80:localhost:8000 serveo.net`
- **Bore:** `bore local 8000 --to bore.pub`

## Troubleshooting

### Error: "ngrok: command not found"

**Solusi:**
1. Pastikan ngrok.exe sudah di-extract
2. Jalankan dari folder ngrok: `cd C:\ngrok` lalu `.\ngrok http 8000`
3. Atau tambahkan ke PATH environment variable

### Error: "authentication failed"

**Solusi:**
```powershell
ngrok config add-authtoken YOUR_AUTH_TOKEN_HERE
```
Get token dari: https://dashboard.ngrok.com/get-started/your-authtoken

### Error: "connection refused"

**Masalah:** Server belum jalan atau salah port

**Solusi:**
1. Pastikan `python app.py` sudah berjalan
2. Cek port nya 8000 (sesuai di app.py)
3. Test server lokal dulu: http://localhost:8000

### Error: "Too many connections"

**Masalah:** Gratis plan limit 40 connections/minute

**Solusi:**
1. Tunggu 1 menit
2. Atau upgrade ke paid plan
3. Atau batasi jumlah user yang akses bersamaan

### Website Loading Lambat

**Penyebab:**
- Server di komputer Anda, bukan cloud
- Speed tergantung internet Anda
- Model inference butuh waktu

**Solusi:**
1. Gunakan internet yang stabil
2. Close aplikasi lain yang pakai bandwidth
3. Test dengan gambar ukuran kecil (<1MB)

### Ngrok Stopped Working

**Penyebab:** Gratis plan timeout setelah 2 jam

**Solusi:**
```powershell
# Stop ngrok: Ctrl+C
# Start lagi:
ngrok http 8000
```
Share URL baru ke teman-teman

## Alternatif Ngrok

### 1. LocalTunnel
```bash
npm install -g localtunnel
lt --port 8000
```
- Pro: Open source, gratis tanpa batas
- Con: Less stable, kadang down

### 2. Serveo
```bash
ssh -R 80:localhost:8000 serveo.net
```
- Pro: Tidak perlu install
- Con: Kadang service down

### 3. Bore
```bash
# Download dari: https://github.com/ekzhang/bore/releases
bore local 8000 --to bore.pub
```
- Pro: Lightweight, cepat
- Con: Masih beta

### 4. Pagekite
```bash
pip install pagekite
pagekite.py 8000 yourname.pagekite.me
```
- Pro: Reliable
- Con: Paid ($3/month)

## Comparison: Ngrok vs Render

| Feature | Ngrok | Render.com |
|---------|-------|------------|
| Setup Time | < 5 min | ~15 min |
| URL Stability | ❌ Berubah setiap restart | ✅ Permanen |
| Komputer Menyala | ❌ Harus | ✅ Tidak perlu |
| HTTPS | ✅ Otomatis | ✅ Otomatis |
| Bandwidth | ⚠️ Terbatas | ✅ Unlimited |
| Cold Start | ✅ Instant | ⚠️ ~30s (free tier) |
| Best For | Demo/Testing | Production/Showcase |

## Rekomendasi

**Gunakan Ngrok jika:**
- ✅ Butuh share **cepat** untuk demo 1-2 hari
- ✅ Hanya untuk **testing** dengan teman
- ✅ **Demo tugas akhir** ke dosen (saat presentasi)
- ✅ Tidak mau ribet setup cloud hosting

**Gunakan Render/Cloud jika:**
- ✅ Butuh **URL permanen** untuk dokumentasi
- ✅ Aplikasi untuk **showcase** jangka panjang
- ✅ Ingin **24/7 online** tanpa komputer menyala
- ✅ Untuk **pengguna banyak** (>10 orang simultan)

## Security Tips

1. **Jangan share API keys** atau sensitive data
2. **Monitor traffic** via ngrok dashboard (http://127.0.0.1:4040)
3. **Use HTTPS** - ngrok sudah default HTTPS
4. **Limit waktu share** - stop ngrok setelah demo selesai
5. **Check who's accessing** via ngrok web interface

## Quick Reference

### Start Server + Ngrok
```powershell
# Terminal 1
cd C:\DogBreedViT
python app.py

# Terminal 2
ngrok http 8000
```

### Stop Everything
```powershell
# Di terminal server: Ctrl+C
# Di terminal ngrok: Ctrl+C
```

### Check Status
```powershell
# Ngrok web dashboard
http://127.0.0.1:4040

# Server health check
http://localhost:8000/health
```

## FAQ

**Q: Apakah gratis selamanya?**
A: Ya, gratis plan permanent. Tapi dengan limitasi (URL berubah, 2 jam timeout).

**Q: Apakah aman?**
A: Ya, ngrok pakai HTTPS encryption. Tapi tetap jangan share data sensitif.

**Q: Bisa diakses dari luar negeri?**
A: Ya! Ngrok global, bisa diakses dari mana saja.

**Q: Berapa lama URL aktif?**
A: Sampai Anda stop ngrok atau timeout (2 jam untuk free plan).

**Q: Bisa multiple users akses bersamaan?**
A: Ya, gratis plan max 40 connections/minute.

**Q: Perlu install Python di semua device?**
A: Tidak! User hanya perlu browser dan internet.

---

## Kesimpulan

**Ngrok adalah solusi TERBAIK untuk:**
- 🎯 **Demo tugas akhir** saat presentasi
- 🎯 **Quick testing** dengan teman
- 🎯 **Share aplikasi** untuk 1-2 hari

**Workflow Recommended:**
1. **Development:** Lokal (localhost:8000)
2. **Demo/Testing:** Ngrok (quick share)
3. **Production/Showcase:** Render.com (permanent URL)

Selamat mencoba! 🚀

---

**Need help?**
- Ngrok Docs: https://ngrok.com/docs
- Ngrok Dashboard: https://dashboard.ngrok.com
- Support: support@ngrok.com
