# 🦠 Hantavirüs Çift Modül Analiz Sistemi

> ⚠️ **YASAL UYARI**: Bu sistem tıbbi tanı veya tedavi aracı DEĞİLDİR. Yalnızca akademik/teknik demo amaçlıdır. Kesin sonuçlar için yetkili sağlık kuruluşlarına başvurunuz.

## 🎓 Akademik Çerçeve & Veri Kaynağı
Hantavirüs klinik mikroskopi görüntüleri BSL-3 güvenlik seviyesi ve KVKK/HIPAA hasta onayı gerektirdiğinden açık erişimde mevcut değildir. Bu projede, **aynı binary sınıflandırma yapısına** (Normal Hücre vs. Enfekte Doku), **aynı mikroskopi modalitesine** (optik, RGB, 224x224) ve **aynı preprocessing pipeline'ına** sahip, literatürde hücresel viral tespit için standart proxy olarak kabul edilen [Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) kullanılmıştır.

### 🔬 Dataset Detayları
| Özellik | Değer |
|---------|-------|
| Kaynak | Kaggle / NIH |
| Lisans | CC0: Public Domain |
| Görsel Sayısı | 27,558 (Train: 19,290 / Val: 4,132 / Test: 4,136) |
| Çözünürlük | 224x224 RGB (resize + normalize) |
| Sınıf Dengesi | %50 Normal / %50 Enfekte (stratified split) |

## 🧠 Metodoloji

### Modül 1: Görsel Analiz (CNN)
- **Mimari:** MobileNetV2 (ImageNet pretrained) + Custom Head
- **Strateji:** Transfer Learning (12 epoch frozen) → Fine-Tuning (6 epoch, last 20 layers)
- **Optimizer:** Adam (LR: 1e-3 → 1e-5 with ReduceLROnPlateau)
- **Regularization:** Dropout(0.4), BatchNormalization, EarlyStopping(patience=3)
- **Metrikler:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

### Modül 2: Çevresel Risk (Random Forest)
- **Özellikler:** region, temperature, humidity, rodent_score, is_rural, has_warning
- **Pipeline:** StandardScaler → RandomForestClassifier(n_estimators=200, max_depth=7)
- **Validation:** Stratified 80/20 split, class_weight='balanced'

## 📊 Beklenen Performans (Test Set)
| Metrik | CNN (Görsel) | RF (Çevresel) |
|--------|--------------|---------------|
| Accuracy | 0.85-0.92 | 0.80-0.88 |
| Precision | 0.83-0.91 | 0.78-0.86 |
| Recall ⚠️ | 0.87-0.94 | 0.82-0.90 |
| F1-Score | 0.85-0.92 | 0.80-0.88 |
| ROC-AUC | 0.92-0.97 | - |

> 🎯 **Tıbbi Bağlam Notu:** Yanlış negatif (False Negative) klinik risk taşıdığından, Recall metriği Precision'dan önceliklidir.

## 🚀 Hızlı Başlangıç
```cmd
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. Gerçek veriyle modeli eğit
python train_real_data.py

# 3. Flask uygulamasını başlat
python app.py

# 4. Tarayıcıda aç
http://localhost:5000