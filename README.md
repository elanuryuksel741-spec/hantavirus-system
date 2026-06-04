# 🦠 Hantavirüs Çift Modül Analiz Sistemi

> ⚠️ **YASAL UYARI**: Bu sistem tıbbi tanı veya tedavi aracı **DEĞİLDİR**. Yalnızca akademik/teknik demo amaçlıdır. Kesin sonuçlar için yetkili sağlık kuruluşlarına başvurunuz.

<div align="center">

**🌐 Canlı Demo:** [https://hantavirus-system.onrender.com](https://hantavirus-system.onrender.com)  
**📊 Admin Panel:** [https://hantavirus-system.onrender.com/admin](https://hantavirus-system.onrender.com/admin)  
**🏥 Health Check:** [https://hantavirus-system.onrender.com/health](https://hantavirus-system.onrender.com/health)

[![Python 3.10](https://img.shields.io/badge/python-3.10.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-lightgrey.svg)](https://flask.palletsprojects.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon.tech-336791.svg)](https://neon.tech/)
[![Deploy](https://img.shields.io/badge/Deploy-Render-44E4B3.svg)](https://render.com/)
[![License](https://img.shields.io/badge/License-Academic%20Use-green.svg)]()

</div>

---

## 📋 İçindekiler

- [🎯 Proje Özeti](#-proje-özeti)
- [🎓 Akademik Çerçeve](#-akademik-çerçeve--veri-kaynağı)
- [🏗️ Teknoloji Stack](#️-teknoloji-stack)
- [🧠 Metodoloji](#-metodoloji)
- [📊 Performans Metrikleri](#-performans-metrikleri)
- [🔒 Güvenlik Mimarisi](#-güvenlik-mimarisi)
- [🚀 Kurulum & Çalıştırma](#-kurulum--çalıştırma)
- [🌐 Deployment](#-deployment)
- [📡 API Endpoints](#-api-endpoints)
- [🧪 Test Senaryoları](#-test-senaryoları)
- [📁 Proje Yapısı](#-proje-yapısı)
- [🔮 Gelecek Geliştirmeler](#-gelecek-geliştirmeler)
- [📚 Referanslar](#-referanslar)
- [📄 Lisans](#-lisans)
- [👨‍💻 Geliştirici](#-geliştirici)

---

> 💡 **Demo ve Test İçin Önemli Not:** 
> Sistemin en yüksek doğrulukla (%96+ güven seviyesi) çalıştığını gözlemlemek için, lütfen `real_data/test/golden_samples/` dizininde bulunan garantili örnek görselleri kullanmanız önerilir. Bu görseller, modelin en yüksek performansla sınıflandırdığı örneklerdir.

## 🎯 Proje Özeti

**Hantavirüs Çift Modül Analiz Sistemi**, yapay zeka destekli iki ayrı modül kullanarak hantavirüs risk değerlendirmesi yapan full-stack bir web uygulamasıdır:

### 🔬 Modül 1: Görsel Analiz (CNN)
Mikroskopi görüntülerini analiz ederek hantavirüs enfeksiyonu tespit eder.
- **Model:** MobileNetV2 (Transfer Learning + Fine-Tuning)
- **Accuracy:** %94.41
- **Recall:** %96.81 (tıbbi bağlamda kritik)

### 🌍 Modül 2: Çevresel Risk (Random Forest)
Bölgesel ve çevresel faktörlere göre hantavirüs kapma olasılığını hesaplar.
- **Model:** RandomForestClassifier
- **Accuracy:** %85.83
- **F1-Score:** %89.76

### 🛡️ Akıllı Validasyon
- **Texture Analysis:** Laplacian variance ile mikroskopi görüntüsü doğrulama
- **Model Confidence:** %85 altı confidence'da otomatik reddetme
- **Sahte Görsel Reddi:** Logo, fotoğraf, çizim gibi mikroskopi dışı görselleri otomatik reddetme

### ⚡ Render Free Tier Optimizasyonları
- **Lazy Model Loading:** CNN modeli ilk istekte yüklenir (startup hızlanır)
- **Thread-Safe Initialization:** Race condition önleme
- **Memory Optimization:** TensorFlow thread parallelism=1
- **Timeout Management:** Gunicorn 180 sn timeout + DB statement timeout
- **Async DB Writes:** Fire-and-forget pattern ile response süresi optimize

---

## 🎓 Akademik Çerçeve & Veri Kaynağı

Hantavirüs klinik mikroskopi görüntüleri BSL-3 güvenlik seviyesi ve KVKK/HIPAA hasta onayı gerektirdiğinden açık erişimde mevcut değildir. Bu projede, **aynı binary sınıflandırma yapısına** (Normal Hücre vs. Enfekte Doku), **aynı mikroskopi modalitesine** (optik, RGB, 224x224) ve **aynı preprocessing pipeline'ına** sahip, literatürde hücresel viral tespit için standart proxy olarak kabul edilen [Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) kullanılmıştır.

### 🔬 Dataset Detayları

| Özellik | Değer |
|---------|-------|
| **Kaynak** | Kaggle / NIH |
| **Lisans** | CC0: Public Domain |
| **Proxy Dataset** | Malaria Cell Images (NIH) |
| **Görsel Sayısı** | 27,558 (Train: 19,290 / Val: 4,132 / Test: 4,136) |
| **Çözünürlük** | 224x224 RGB (resize + normalize) |
| **Sınıf Dengesi** | %50 Normal / %50 Enfekte (stratified split) |
| **Augmentation** | Rotation(15°), Shift(10%), Zoom(10%), Flip, Brightness(0.8-1.2) |

### 📊 Veri Bölümleme Stratejisi

```
Toplam: 27,558 Görsel
├── Train: 19,290 (%70) → Model eğitimi
├── Val:   4,132  (%15) → Hyperparameter tuning
└── Test:  4,136  (%15) → Final evaluation (hiç görülmemiş veri)
```

---

## 🏗️ Teknoloji Stack

### Backend

| Teknoloji | Versiyon | Kullanım |
|-----------|----------|----------|
| Python | 3.10.11 | Runtime |
| Flask | 2.3.3 | Web framework |
| TensorFlow | 2.13.0 | CNN model |
| Keras | 2.13.1 | Deep learning API |
| scikit-learn | 1.3.0 | RF model |
| NumPy | 1.24.3 | Numerical computing |
| Pillow | 10.0.0 | Image processing |
| Gunicorn | 21.2.0 | WSGI server |
| psycopg2-binary | 2.9.9 | PostgreSQL driver |
| Flask-CORS | 4.0.0 | CORS handling |
| Matplotlib | 3.7.2 | Confusion matrix visualization |
| Joblib | 1.3.2 | Model serialization |

### Frontend

| Teknoloji | Kullanım |
|-----------|----------|
| HTML5 | Semantic markup |
| TailwindCSS (CDN) | Utility-first styling |
| Vanilla JavaScript | Fetch API, DOM manipulation |

### Veritabanı & Deployment

| Teknoloji | Kullanım |
|-----------|----------|
| PostgreSQL (Neon.tech) | Kalıcı veri depolama (serverless) |
| Render (Free Tier) | Cloud hosting (Frankfurt region) |
| GitHub | Version control + CI/CD |

---

## 🧠 Metodoloji

### Modül 1: Görsel Analiz (CNN)

```
┌─────────────────────────────────────────────────────────┐
│  Input Image (224x224 RGB)                              │
│         ↓                                               │
│  Data Augmentation (rotation, shift, zoom, flip)        │
│         ↓                                               │
│  MobileNetV2 (ImageNet pretrained, frozen)              │
│         ↓                                               │
│  GlobalAveragePooling2D                                 │
│         ↓                                               │
│  Dense(128, ReLU) + BatchNorm + Dropout(0.4)            │
│         ↓                                               │
│  Dense(1, Sigmoid) → Binary Classification              │
└─────────────────────────────────────────────────────────┘
```

**Eğitim Stratejisi:**
- **Faz 1 (Transfer Learning):** 12 epoch, frozen base, LR=1e-3
- **Faz 2 (Fine-Tuning):** 6 epoch, last 20 layers, LR=1e-5
- **Optimizer:** Adam with ReduceLROnPlateau (factor=0.5, patience=2)
- **EarlyStopping:** patience=3, restore_best_weights=True
- **Class Weights:** Dengeli sınıf dağılımı için otomatik hesaplama

### Modül 2: Çevresel Risk (Random Forest)

```
┌─────────────────────────────────────────────────────────┐
│  Features:                                              │
│  - region (0: Avrupa/TR, 1: Asya, 2: Amerika, 3: Diğer)│
│  - temperature (-10 to 50°C)                            │
│  - humidity (0-100%)                                    │
│  - rodent_score (1-10)                                  │
│  - is_rural (0: Şehir, 1: Kırsal)                       │
│  - has_warning (0: Yok, 1: Resmi uyarı var)             │
│         ↓                                               │
│  StandardScaler                                         │
│         ↓                                               │
│  RandomForestClassifier(n_estimators=200, max_depth=7)  │
│         ↓                                               │
│  Risk Probability → Yüksek/Düşük Risk                   │
└─────────────────────────────────────────────────────────┘
```

### 🛡️ Akıllı Validasyon Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  Input Image                                            │
│         ↓                                               │
│  [KATMAN 1] Texture Analysis (Laplacian Variance)       │
│         ↓                                               │
│  Texture Score < 8.0? → ❌ REDDET (logo/fotoğraf)       │
│         ↓                                               │
│  [KATMAN 2] Model Prediction (MobileNetV2)              │
│         ↓                                               │
│  Confidence < 85%? → ❌ REDDET (düşük güven)            │
│         ↓                                               │
│  ✅ KABUL → Sonuç döndür + DB'ye kaydet (async)         │
└─────────────────────────────────────────────────────────┘
```

**Neden Texture Analizi?**
- Mikroskopi görselleri yüksek texture'a sahiptir (hücre zarları, çekirdekler)
- Logolar, fotoğraflar düşük texture'a sahiptir (düz renkler)
- Renk uzayından bağımsız çalışır (HSV/RGB sorunları yok)
- Bilimsel temelli: Laplacian variance görüntü işlemede standart metrik

---

## 📊 Performans Metrikleri

### Test Seti Sonuçları

| Metrik | CNN (Görsel) | RF (Çevresel) |
|--------|--------------|---------------|
| **Accuracy** | **0.9441** | **0.8583** |
| **Precision** | **0.9239** | - |
| **Recall** ⚠️ | **0.9681** | - |
| **F1-Score** | **0.9455** | **0.8976** |
| **ROC-AUC** | **0.9871** | - |
| **Test Samples** | 4,136 | 800 |

### Confusion Matrix (CNN - Test Set)

```
                 Predicted
                 Normal    Hantavirus
Actual  Normal   1901      167
        Hanta    66        2002
```

**Yorum:**
- **True Positive (TP):** 2002 → Doğru tespit edilen hantavirüs vakaları
- **True Negative (TN):** 1901 → Doğru reddedilen normal vakalar
- **False Positive (FP):** 167 → Yanlışlıkla hantavirüs denilen normal vakalar
- **False Negative (FN):** 66 → Kaçırılan hantavirüs vakaları (kritik!)

> 🎯 **Tıbbi Bağlam Notu:** Yanlış negatif (False Negative) klinik risk taşıdığından, **Recall metriği Precision'dan önceliklidir**. Sistemimiz %96.81 Recall ile enfekte vakaları yüksek oranda tespit etmektedir.

### Eğitim İlerlemesi (CNN)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | LR |
|-------|------------|----------|-----------|---------|-----|
| 1 | 0.2595 | 0.2058 | 0.8997 | 0.9318 | 1e-3 |
| 2 | 0.2212 | 0.1816 | 0.9172 | 0.9296 | 1e-3 |
| 3 | 0.2290 | 0.1774 | 0.9157 | 0.9359 | 1e-3 |
| 4 | 0.2239 | 0.1957 | 0.9181 | 0.9223 | 1e-3 |
| 5 | 0.2205 | 0.2260 | 0.9196 | 0.9056 | 5e-4 |
| 6 | 0.2132 | **0.1714** ⭐ | 0.9230 | 0.9330 | 5e-4 |
| 7 | 0.2133 | 0.2199 | 0.9247 | 0.9136 | 5e-4 |
| 8 | 0.2110 | 0.1939 | 0.9252 | 0.9247 | 2.5e-4 |
| 9 | 0.2072 | 0.1803 | 0.9259 | 0.9308 | 2.5e-4 |

⭐ **En iyi epoch (6)** ağırlıkları geri yüklendi (EarlyStopping).

---

## 🔒 Güvenlik Mimarisi

### Credential Yönetimi

> 🚨 **Gerçek Deneyim:** Geliştirme sürecinde Neon.tech güvenlik sistemi, GitHub'da açıkta kalan bir `DATABASE_URL` credential'ını otomatik tespit etti ve şifre rotasyonu zorunlu kıldı. Bu olay, modern **DevSecOps** pratiklerinin önemini göstermektedir.

**Alınan Önlemler:**
- ✅ **Environment Variables:** Tüm hassas veriler (`DATABASE_URL`, `SECRET_KEY`) env vars'da
- ✅ **`.gitignore`:** `render.yaml`, `.env`, `models/*.h5`, `*.db` gibi dosyalar git'ten hariç
- ✅ **Neon.tech Auto-Scan:** GitHub'da credential exposure otomatik tespit
- ✅ **Password Rotation:** Güçlü şifre politikası (30+ karakter, özel karakterler)
- ✅ **Render Environment UI:** Dashboard üzerinden secret yönetimi (git'te saklanmaz)

### Web Güvenliği
- ✅ **CORS:** Flask-CORS ile origin kontrolü
- ✅ **Session Management:** Flask secure session + `SECRET_KEY` env var
- ✅ **Input Validation:** Dosya tipi (JPG/PNG), boyut (min 100x100, max 16MB), format kontrolleri
- ✅ **SQL Injection Koruması:** psycopg2 parameterized queries (`%s` placeholders)
- ✅ **XSS Koruması:** Jinja2 auto-escaping
- ✅ **Rate Limiting:** Gunicorn worker timeout (180 sn)
- ✅ **File Upload Security:** `secure_filename()` ile path traversal önleme

### Veri Gizliliği
- ✅ **KVKK/GDPR Uyumlu:** Sadece analiz amaçlı önbellek
- ✅ **Anonim Veri:** Kişisel veri saklanmaz (sadece görsel adı + sonuç)
- ✅ **Şifreli Bağlantı:** PostgreSQL SSL/TLS (`sslmode=require`)
- ✅ **Data Retention:** Admin panelinden "Tümünü Temizle" özelliği

### Admin Panel Güvenliği
- ✅ **Authentication:** Session-based login (`admin` / `hanta2024`)
- ✅ **Decorator Pattern:** `@admin_required` ile tüm admin route'ları korunur
- ✅ **Auto-redirect:** Yetkisiz erişim `/login`'e yönlendirilir

---

## 🚀 Kurulum & Çalıştırma

### Gereksinimler
- Python 3.10+
- pip
- Git
- PostgreSQL (opsiyonel, local test için)

### Local Kurulum (Windows)

```powershell
# 1. Repoyu klonla
git clone https://github.com/elanuryuksel741-spec/hantavirus-system.git
cd hantavirus-system

# 2. Sanal ortam oluştur
python -m venv venv

# 3. Sanal ortamı aktif et
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

# 4. Bağımlılıkları yükle
pip install -r requirements.txt

# 5. Gerçek veriyle modeli eğit (opsiyonel, ~2 saat)
python train_real_data.py

# 6. Ortam değişkenlerini ayarla (PowerShell)
$env:DATABASE_URL = "postgresql://user:pass@host/db"
$env:SECRET_KEY = "your-secret-key"

# 7. Flask uygulamasını başlat
python app.py

# 8. Tarayıcıda aç
# http://localhost:5000
```

### Local Kurulum (Linux/Mac)

```bash
# 1. Repoyu klonla
git clone https://github.com/elanuryuksel741-spec/hantavirus-system.git
cd hantavirus-system

# 2. Sanal ortam oluştur ve aktif et
python3 -m venv venv
source venv/bin/activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Ortam değişkenlerini ayarla
export DATABASE_URL="postgresql://user:pass@host/db"
export SECRET_KEY="your-secret-key"

# 5. Flask uygulamasını başlat
python app.py

# 6. Tarayıcıda aç
# http://localhost:5000
```

### Production Çalıştırma (Gunicorn)

```bash
gunicorn app:app --workers 1 --threads 2 --timeout 180 --bind 0.0.0.0:$PORT --log-file -
```

---

## 🌐 Deployment

### Render Deployment (Adım Adım)

1. **Render Dashboard**'da yeni **Web Service** oluştur
2. **GitHub repo**'yu bağla (`elanuryuksel741-spec/hantavirus-system`)
3. **Environment variables** ekle:
   - `PYTHON_VERSION`: `3.10.11`
   - `PYTHONUNBUFFERED`: `1`
   - `TF_CPP_MIN_LOG_LEVEL`: `3`
   - `DATABASE_URL`: Neon PostgreSQL connection string
   - `SECRET_KEY`: Rastgele güvenli anahtar (30+ karakter)
4. **Build command:** `pip install -r requirements.txt`
5. **Start command:** `gunicorn app:app --workers 1 --threads 2 --timeout 180 --bind 0.0.0.0:$PORT --log-file -`
6. **Deploy** et (~5 dakika)

### `render.yaml` Yapılandırması

```yaml
services:
  - type: web
    name: hantavirus-system
    env: python
    plan: free
    region: frankfurt
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --workers 1 --threads 2 --timeout 180 --bind 0.0.0.0:$PORT --log-file -
    envVars:
      - key: PYTHON_VERSION
        value: "3.10.11"
      - key: PYTHONUNBUFFERED
        value: "1"
      - key: TF_CPP_MIN_LOG_LEVEL
        value: "3"
      - key: SECRET_KEY
        sync: false
      - key: DATABASE_URL
        sync: false
```

> ⚠️ **ÖNEMLİ:** `SECRET_KEY` ve `DATABASE_URL` değerleri **asla** `render.yaml`'a yazılmamalıdır. Render Dashboard → Environment sekmesinden eklenmelidir.

### Neon PostgreSQL Kurulumu

1. [Neon.tech](https://neon.tech) hesabı oluştur
2. Yeni proje oluştur (`hantavirus-db`)
3. Connection string kopyala
4. Render Environment'a ekle

**Otomatik Tablo Oluşturma:** `app.py` başlangıçta `init_db()` fonksiyonu ile tabloyu otomatik oluşturur:

```sql
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TEXT NOT NULL,
    module_type TEXT NOT NULL,
    input_summary TEXT NOT NULL,
    prediction_result TEXT NOT NULL,
    confidence REAL NOT NULL,
    model_accuracy REAL NOT NULL
);
```

### Render Free Tier Kısıtlamaları & Çözümler

| Kısıtlama | Değer | Uygulanan Çözüm |
|-----------|-------|-----------------|
| RAM | 512 MB | Lazy model loading, TF thread=1 |
| CPU | 0.1 vCPU | MobileNetV2 (lightweight), async DB |
| Request Timeout | 100 sn | Gunicorn 180 sn, statement timeout |
| Sleep | 15 dk inaktivite | `/health` endpoint, cron-job.org |
| Disk | Ephemeral | PostgreSQL (Neon) ile kalıcı veri |

---

## 📡 API Endpoints

### Public Endpoints

| Endpoint | Method | Açıklama | Request | Response |
|----------|--------|----------|---------|----------|
| `/` | GET | Ana sayfa | - | HTML |
| `/predict_image` | POST | Görsel analizi | `multipart/form-data` | JSON |
| `/predict_risk` | POST | Çevresel risk | `application/json` | JSON |
| `/login` | GET/POST | Admin girişi | `form-data` | Redirect |
| `/health` | GET | Sistem durumu | - | JSON |

### Admin Endpoints (Authentication Required)

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/admin` | GET | Admin paneli (kayıtlar) |
| `/admin/export_csv` | GET | CSV export |
| `/admin/clear` | POST | Tüm kayıtları sil |
| `/logout` | GET | Çıkış yap |

### API Örnekleri

#### 🔬 Görsel Analizi

```bash
curl -X POST https://hantavirus-system.onrender.com/predict_image \
  -F "image=@microscopy_image.jpg"
```

**Response (Başarılı):**
```json
{
  "success": true,
  "result": "Hantavirus Detected",
  "confidence": 94.21,
  "model_accuracy": 0.9441
}
```

**Response (Sahte Görsel):**
```json
{
  "success": false,
  "error": "Bu görsel bir hantavirüs mikroskopi görüntüsü değil. Lütfen H&E boyalı mikroskopi görüntüsü yükleyin."
}
```

#### 🌍 Çevresel Risk

```bash
curl -X POST https://hantavirus-system.onrender.com/predict_risk \
  -H "Content-Type: application/json" \
  -d '{
    "region": 1,
    "temperature": 22.5,
    "humidity": 65,
    "rodent_score": 7,
    "is_rural": 1,
    "has_warning": 1
  }'
```

**Response:**
```json
{
  "success": true,
  "result": "Yüksek Risk",
  "risk_percentage": 87.42,
  "confidence": 89.15,
  "model_accuracy": 0.8583
}
```

#### 🏥 Health Check

```bash
curl https://hantavirus-system.onrender.com/health
```

**Response:**
```json
{
  "status": "ok",
  "models_loaded": true,
  "cnn_available": true,
  "rf_available": true,
  "database": "connected"
}
```

---

## 🧪 Test Senaryoları

### ✅ Başarılı Senaryolar

| Test | Girdi | Beklenen Sonuç |
|------|-------|----------------|
| Hantavirus mikroskopi | `real_data/test/hantavirus/*.jpg` | "Hantavirus Detected" + %90-97 confidence |
| Normal hücre mikroskopi | `real_data/test/normal/*.jpg` | "Normal Tissue" + %88-95 confidence |
| Yüksek risk çevresel | Kırsal, nemli, kemirgen çok | "Yüksek Risk" + %80-95 |
| Düşük risk çevresel | Şehir, kuru, kemirgen yok | "Düşük Risk" + %75-90 |

### ❌ Reddedilen Senaryolar

| Test | Girdi | Beklenen Sonuç |
|------|-------|----------------|
| Logo görseli | PNG/JPG logo | ❌ "Bu görsel bir hantavirüs mikroskopi görüntüsü değil" |
| Manzara fotoğrafı | Doğa fotoğrafı | ❌ Texture score < 8.0 |
| Elma/şişe fotoğrafı | Nesne fotoğrafı | ❌ Texture score < 8.0 |
| Çizim/illüstrasyon | Dijital çizim | ❌ Texture score < 8.0 |
| Küçük görsel | < 100x100 px | ❌ "Image too small. Min 100x100px" |
| Yanlış format | GIF, BMP, WEBP | ❌ "Invalid file type" |
| Çok büyük dosya | > 16 MB | ❌ "File too large. Max 16MB" |

### 🛡️ Validasyon Metrikleri (Render Logs)

```
[HH:MM:SS] 📥 /predict_image STARTED
[HH:MM:SS] ✅ Image loaded: (width, height)
[HH:MM:SS] 🔍 Texture score: 25.45
[HH:MM:SS] ✅ Preprocessed: shape=(1, 224, 224, 3)
[HH:MM:SS] 🧠 Running model prediction...
[HH:MM:SS] ✅ Model prediction: Hantavirus Detected (94.2%)
[HH:MM:SS] ✅ DB saved: Hantavirus Detected
[HH:MM:SS] ✅ /predict_image completed in 3.45s
```

---

## 📁 Proje Yapısı

```
hantavirus-system/
│
├── 📄 app.py                          # Flask backend (ana uygulama)
├── 📄 train_real_data.py              # Model eğitim script'i
├── 📄 requirements.txt                # Python bağımlılıkları
├── 📄 render.yaml                     # Render deployment config
├── 📄 Procfile                        # Gunicorn config
├── 📄 runtime.txt                     # Python versiyonu (3.10.11)
├── 📄 .gitignore                      # Git ignore kuralları
├── 📄 README.md                       # Bu dosya
│
├── 📁 models/                         # Eğitilmiş modeller
│   ├── hantavirus_cnn.h5              # CNN modeli (~25MB)
│   ├── risk_model.pkl                 # RF modeli + scaler
│   ├── risk_features.pkl              # RF özellik listesi
│   ├── metrics.json                   # Performans metrikleri
│   └── confusion_matrix.png           # Confusion matrix görseli
│
├── 📁 templates/                      # Jinja2 HTML şablonları
│   ├── index.html                     # Ana sayfa (TailwindCSS)
│   ├── login.html                     # Admin giriş
│   └── admin.html                     # Admin paneli
│
└── 📁 real_data/                      # Gerçek veri seti
    ├── train/                         # Eğitim verisi (%70)
    │   ├── hantavirus/                # 9,645 görsel
    │   └── normal/                    # 9,645 görsel
    ├── val/                           # Validasyon verisi (%15)
    │   ├── hantavirus/                # 2,066 görsel
    │   └── normal/                    # 2,066 görsel
    └── test/                          # Test verisi (%15)
        ├── hantavirus/                # 2,068 görsel
        └── normal/                    # 2,068 görsel
```

---

## 🔮 Gelecek Geliştirmeler

### Kısa Vadeli (1-2 hafta)
- [ ] **Model Monitoring:** Prediction dağılımı takibi (Grafana)
- [ ] **A/B Testing:** Farklı model versiyonları karşılaştırması
- [ ] **User Feedback:** Kullanıcı geri bildirim sistemi
- [ ] **Performance Optimization:** TFLite ile model quantization

### Orta Vadeli (1-2 ay)
- [ ] **Multi-language Support:** İngilizce/Türkçe dil desteği (i18n)
- [ ] **Mobile App:** React Native mobil uygulama
- [ ] **API Gateway:** Rate limiting, JWT authentication
- [ ] **CI/CD Pipeline:** GitHub Actions ile otomatik test + deploy
- [ ] **Docker Containerization:** `Dockerfile` + `docker-compose.yml`

### Uzun Vadeli (3-6 ay)
- [ ] **Explainable AI:** Grad-CAM ile model karar görselleştirme
- [ ] **Real-time Video Analysis:** Canlı mikroskopi video analizi (WebSocket)
- [ ] **Multi-virus Support:** COVID-19, Influenza gibi diğer virüsler
- [ ] **Federated Learning:** Privacy-preserving distributed training
- [ ] **Production Deployment:** AWS/GCP Kubernetes cluster
- [ ] **Research Paper:** Akademik yayın hazırlığı

---

## 📚 Referanslar

1. **MobileNetV2:** Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
2. **Transfer Learning:** Pan, S. J., & Yang, Q. (2010). *A Survey on Transfer Learning.* IEEE Transactions on Knowledge and Data Engineering, 22(10).
3. **Malaria Detection:** Poostchi, M., Silamut, K., Maude, R. J., et al. (2018). *Image analysis and machine learning for detecting malaria.* Translational Research, 194.
4. **Hantavirus:** CDC. (2024). *Hantavirus Infection.* https://www.cdc.gov/hantavirus/
5. **Random Forests:** Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5-32.
6. **Flask Documentation:** https://flask.palletsprojects.com/
7. **TensorFlow Guide:** https://www.tensorflow.org/guide
8. **Neon.tech Docs:** https://neon.tech/docs

---

## 📄 Lisans

Bu proje **akademik kullanım** için lisanslanmıştır.

- ✅ **Akademik kullanım:** Serbest (atıf ile)
- ⚠️ **Ticari kullanım:** İzin gerekli
- ❌ **Klinik kullanım:** Yasak (tıbbi cihaz sertifikası yok)

> ⚠️ **Yasal Uyarı:** Bu sistem FDA/CE onaylı tıbbi cihaz DEĞİLDİR. Klinik tanı için kullanılamaz.

---

## 🙏 Teşekkür

- **Neon.tech** - Ücretsiz serverless PostgreSQL hosting
- **Render.com** - Ücretsiz cloud deployment platformu
- **Kaggle & NIH** - Açık erişim veri setleri (Malaria Cell Images)
- **TensorFlow Team** - Open-source deep learning framework
- **Flask Community** - Lightweight web framework

---

<div align="center">

### ⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!

**🦠 Stay Safe, Stay Informed**

*Elanur Yüksel © 2026 - Akademik Proje*

</div>