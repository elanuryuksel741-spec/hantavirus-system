#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hantavirus Dual Module Analysis System - Setup & Training Script
Generates synthetic data and trains both CNN (MobileNetV2) and Random Forest models.
⚠️ WARNING: This system is NOT for clinical diagnosis. For educational/demo purposes only.
"""

import os
import json
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('data/synthetic_images/train/hantavirus', exist_ok=True)
os.makedirs('data/synthetic_images/train/normal', exist_ok=True)
os.makedirs('data/synthetic_images/val/hantavirus', exist_ok=True)
os.makedirs('data/synthetic_images/val/normal', exist_ok=True)
os.makedirs('data/synthetic_images/test/hantavirus', exist_ok=True)
os.makedirs('data/synthetic_images/test/normal', exist_ok=True)

print("🔬 Generating synthetic microscopy images...")

def generate_hantavirus_image(size=224):
    """Generate synthetic hantavirus microscopy image with viral particle patterns."""
    img = Image.new('RGB', (size, size), color=(240, 245, 250))
    draw = ImageDraw.Draw(img)
    
    # Background cellular texture
    for _ in range(50):
        x, y = np.random.randint(0, size, 2)
        r = np.random.randint(3, 15)
        shade = np.random.randint(200, 230)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(shade, shade+10, shade+20))
    
    # Viral particles (circular clusters with halo effect)
    num_particles = np.random.randint(8, 25)
    for _ in range(num_particles):
        cx, cy = np.random.randint(20, size-20, 2)
        radius = np.random.randint(6, 18)
        
        # Core
        draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], 
                    fill=(180, 60, 80, 200))
        
        # Halo effect
        for r in range(radius+2, radius+8, 2):
            alpha = max(20, 80 - r*5)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], 
                        outline=(220, 100, 120, alpha), width=1)
    
    # Add slight noise/blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    return img

def generate_normal_image(size=224):
    """Generate synthetic normal tissue microscopy image with regular cellular patterns."""
    img = Image.new('RGB', (size, size), color=(245, 250, 248))
    draw = ImageDraw.Draw(img)
    
    # Regular cellular structure (hexagonal-like patterns)
    for row in range(0, size, 28):
        for col in range(0, size, 28):
            if np.random.random() > 0.1:  # 90% cells present
                cx, cy = col + np.random.randint(-3, 4), row + np.random.randint(-3, 4)
                r = np.random.randint(10, 16)
                shade = np.random.randint(210, 235)
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], 
                           fill=(shade, shade+5, shade-10))
                # Cell nucleus
                draw.ellipse([cx-4, cy-4, cx+4, cy+4], 
                           fill=(shade-40, shade-30, shade-50))
    
    # Subtle texture lines
    for _ in range(30):
        x1, y1 = np.random.randint(0, size, 2)
        x2, y2 = x1 + np.random.randint(-15, 16), y1 + np.random.randint(-15, 16)
        draw.line([x1, y1, x2, y2], fill=(200, 210, 205), width=1)
    
    img = img.filter(ImageFilter.GaussianBlur(radius=0.2))
    return img

def save_dataset_images(num_per_class=600):
    """Generate and save synthetic images for train/val/test splits."""
    splits = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    for class_name, gen_func in [('hantavirus', generate_hantavirus_image), 
                                  ('normal', generate_normal_image)]:
        images = [gen_func() for _ in range(num_per_class)]
        
        np.random.shuffle(images)
        train_end = int(num_per_class * splits['train'])
        val_end = train_end + int(num_per_class * splits['val'])
        
        for i, img in enumerate(images[:train_end]):
            img.save(f'data/synthetic_images/train/{class_name}/syn_{i:04d}.png')
        for i, img in enumerate(images[train_end:val_end]):
            img.save(f'data/synthetic_images/val/{class_name}/syn_{i:04d}.png')
        for i, img in enumerate(images[val_end:]):
            img.save(f'data/synthetic_images/test/{class_name}/syn_{i:04d}.png')
    
    print(f"✅ Generated {num_per_class*2} synthetic images (600/600 per class)")

save_dataset_images(600)

print("🧠 Training CNN model (MobileNetV2 transfer learning)...")

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/synthetic_images/train',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    'data/synthetic_images/val',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

# Build MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

cnn_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(name='precision')]
)

# Train with early stopping
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = cnn_model.fit(
    train_generator,
    epochs=8,
    validation_data=val_generator,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate on test set
test_generator = val_datagen.flow_from_directory(
    'data/synthetic_images/test',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)
cnn_metrics = cnn_model.evaluate(test_generator, verbose=0)
cnn_accuracy = cnn_metrics[1]
cnn_precision = cnn_metrics[2]

cnn_model.save('models/hantavirus_cnn.h5')
print(f"✅ CNN saved. Test Accuracy: {cnn_accuracy:.4f}, Precision: {cnn_precision:.4f}")

print("🌲 Training Random Forest model for environmental risk...")

def generate_environmental_data(n_samples=800):
    """Generate synthetic environmental data with biologically plausible patterns."""
    np.random.seed(42)
    data = []
    
    for _ in range(n_samples):
        region = np.random.choice([0, 1, 2, 3], p=[0.35, 0.30, 0.20, 0.15])
        temperature = np.random.uniform(5, 35)
        humidity = np.random.uniform(20, 95)
        rodent_score = np.random.randint(1, 11)
        is_rural = np.random.choice([0, 1], p=[0.6, 0.4])
        has_warning = np.random.choice([0, 1], p=[0.85, 0.15])
        
        # Risk calculation (biologically inspired logic)
        risk_score = 0
        risk_score += 0.3 if region in [1, 2] else 0  # Asia/Americas higher risk
        risk_score += 0.25 if 15 <= temperature <= 28 else 0  # Optimal temp range
        risk_score += 0.2 if humidity > 60 else 0
        risk_score += rodent_score * 0.04
        risk_score += 0.35 if is_rural == 1 else 0
        risk_score += 0.4 if has_warning == 1 else 0
        
        # Add noise and threshold
        risk_score += np.random.normal(0, 0.15)
        risk_level = 1 if risk_score > 0.7 else 0
        
        data.append([region, temperature, humidity, rodent_score, is_rural, has_warning, risk_level])
    
    return np.array(data)

rf_data = generate_environmental_data(800)
X_rf = rf_data[:, :6]
y_rf = rf_data[:, 6]

# Split and scale
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_rf)
X_test_scaled = scaler.transform(X_test_rf)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=150, max_depth=6, random_state=42, n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train_rf)

rf_accuracy = rf_model.score(X_test_scaled, y_test_rf)
print(f"✅ Random Forest saved. Test Accuracy: {rf_accuracy:.4f}")

# Save models and scalers
joblib_path = 'models/risk_model.pkl'
with open(joblib_path, 'wb') as f:
    pickle.dump({'model': rf_model, 'scaler': scaler}, f)

with open('models/risk_features.pkl', 'wb') as f:
    pickle.dump(['region', 'temperature', 'humidity', 'rodent_score', 'is_rural', 'has_warning'], f)

# Save metrics
metrics = {
    "cnn_accuracy": round(float(cnn_accuracy), 4),
    "cnn_precision": round(float(cnn_precision), 4),
    "rf_accuracy": round(float(rf_accuracy), 4),
    "test_samples": int(len(y_test_rf) + len(test_generator.filenames))
}
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("🎉 Setup complete! Models saved to 'models/' folder.")
print(f"📊 Final Metrics: {json.dumps(metrics, indent=2)}")
print("\n⚠️  LEGAL NOTICE: This system is for educational/demo purposes only.")
print("   It is NOT a medical diagnostic tool. Consult healthcare professionals for real diagnoses.")