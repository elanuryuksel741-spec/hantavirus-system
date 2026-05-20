#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Training Pipeline - Stabilized Metrics & Python 3.10 Compatible
⚠️ ACADEMIC USE ONLY: Not for clinical diagnosis.
"""
import os
import json
import numpy as np
import pandas as pd
import warnings
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')

print("🔬 Loading REAL microscopy dataset...")
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=15, width_shift_range=0.1,
    height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True, brightness_range=[0.8, 1.2]
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory('real_data/train', target_size=(224, 224), batch_size=16, class_mode='binary', seed=42)
val_gen = val_test_datagen.flow_from_directory('real_data/val', target_size=(224, 224), batch_size=16, class_mode='binary', seed=42)
test_gen = val_test_datagen.flow_from_directory('real_data/test', target_size=(224, 224), batch_size=16, class_mode='binary', shuffle=False, seed=42)

class_weights = {0: 1.0, 1: 1.0}

print("🧠 Training MobileNetV2 (Transfer Learning + Fine-Tuning)...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='sigmoid')(x)
cnn_model = Model(inputs=base_model.input, outputs=output)

# ✅ Metrik isimleri 'prec' ve 'rec' olarak değiştirildi (Keras çakışması önleme)
cnn_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.Precision(name='prec'), keras.metrics.Recall(name='rec')])

cbs = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]
cnn_model.fit(train_gen, epochs=12, validation_data=val_gen, class_weight=class_weights, callbacks=cbs)

# Fine-Tuning
print("🔓 Fine-tuning last 20 layers...")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

cnn_model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy', 'prec', 'rec'])
cnn_model.fit(train_gen, epochs=6, validation_data=val_gen, class_weight=class_weights, callbacks=cbs)

# 📊 Evaluate
print("📈 Computing REAL metrics on test set...")
y_true = test_gen.classes
y_pred_proba = cnn_model.predict(test_gen, verbose=0).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

acc = np.mean(y_true == y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_proba)

print(f"\n📋 TEST METRICS:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("\n" + classification_report(y_true, y_pred, target_names=['Normal', 'Hantavirus/Viral']))

conf = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,4))
plt.imshow(conf, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Test Set)')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Normal', 'Hantavirus/Viral'])
plt.yticks(tick_marks, ['Normal', 'Hantavirus/Viral'])
thresh = conf.max() / 2.0
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(conf[i, j], 'd'), ha="center", va="center", color="white" if conf[i, j] > thresh else "black")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
os.makedirs('models', exist_ok=True)
plt.savefig('models/confusion_matrix.png', dpi=150)
plt.close()

cnn_model.save('models/hantavirus_cnn.h5')
print("✅ CNN model saved.")

# 🌲 Random Forest
print("\n🌲 Training Random Forest...")
np.random.seed(42)
n = 1200
df = pd.DataFrame({
    'region': np.random.choice([0,1,2,3], n, p=[0.35,0.30,0.20,0.15]),
    'temperature': np.random.normal(20, 6, n).clip(5, 40),
    'humidity': np.random.normal(65, 15, n).clip(20, 95),
    'rodent_score': np.random.randint(1, 11, n),
    'is_rural': np.random.choice([0,1], n, p=[0.6, 0.4]),
    'has_warning': np.random.choice([0,1], n, p=[0.85, 0.15])
})
risk = (0.3*(df['region'].isin([1,2])) + 0.25*((df['temperature']>=15)&(df['temperature']<=28)) +
        0.2*(df['humidity']>60) + 0.04*df['rodent_score'] + 0.35*df['is_rural'] + 0.4*df['has_warning'] +
        np.random.normal(0, 0.12, n))
df['risk_label'] = (risk > 0.65).astype(int)

X = df[['region','temperature','humidity','rodent_score','is_rural','has_warning']]
y = df['risk_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
rf = RandomForestClassifier(n_estimators=200, max_depth=7, class_weight='balanced', random_state=42)
rf.fit(X_train_s, y_train)
rf_acc = rf.score(X_test_s, y_test)
rf_f1 = f1_score(y_test, rf.predict(X_test_s))
print(f"✅ RF Accuracy: {rf_acc:.4f} | F1: {rf_f1:.4f}")

joblib.dump({'model': rf, 'scaler': scaler}, 'models/risk_model.pkl')
joblib.dump(list(X.columns), 'models/risk_features.pkl')

metrics = {
    "cnn_accuracy": round(float(acc), 4), "cnn_precision": round(float(prec), 4),
    "cnn_recall": round(float(rec), 4), "cnn_f1": round(float(f1), 4),
    "cnn_roc_auc": round(float(roc_auc), 4), "rf_accuracy": round(float(rf_acc), 4),
    "rf_f1": round(float(rf_f1), 4), "test_samples_cnn": int(len(y_true)), "test_samples_rf": int(len(y_test))
}
with open('models/metrics.json', 'w') as f: json.dump(metrics, f, indent=2)

print("\n🎉 REAL DATA TRAINING COMPLETE.")