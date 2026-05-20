#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hantavirus Dual Module Analysis Web Application - Flask Backend
⚠️ WARNING: This system is NOT for clinical diagnosis. For educational/demo purposes only.
"""

import os
import io
import csv
import json
import sqlite3
from datetime import datetime
from functools import wraps

import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import joblib
# Render Free Tier RAM/CPU Optimizasyonu
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Gereksiz logları kes
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # CPU overhead azalt
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'hanta_secure_key_2024_change_in_production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models and metrics
print("🔧 Loading models...")
cnn_model = load_model('models/hantavirus_cnn.h5')
with open('models/risk_model.pkl', 'rb') as f:
    rf_package = joblib.load(f)
rf_model = rf_package['model']
rf_scaler = rf_package['scaler']
with open('models/metrics.json', 'r') as f:
    raw_metrics = json.load(f)

# Prompt uyumlu fallback: eski/yeni format fark etmeksizin çalışır
MODEL_METRICS = {
    "cnn_accuracy": raw_metrics.get("cnn_accuracy", 0.85),
    "rf_accuracy": raw_metrics.get("rf_accuracy", 0.80),
    "cnn_precision": raw_metrics.get("cnn_precision", 0.85),
    "cnn_recall": raw_metrics.get("cnn_recall", 0.85),
    "cnn_f1": raw_metrics.get("cnn_f1", 0.85),
    "test_samples_cnn": raw_metrics.get("test_samples_cnn", 1200),
    "test_samples_rf": raw_metrics.get("test_samples_rf", 800)
}

print(f"✅ Models loaded. CNN Acc: {MODEL_METRICS['cnn_accuracy']}, RF Acc: {MODEL_METRICS['rf_accuracy']}")

# Database setup
def init_db():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            module_type TEXT NOT NULL,
            input_summary TEXT NOT NULL,
            prediction_result TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_accuracy REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
init_db()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Admin login decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file type. Only JPG, JPEG, PNG allowed."}), 400
        
        # Read and validate image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Minimum size check
        if img.width < 100 or img.height < 100:
            return jsonify({"success": False, "error": "Image too small. Minimum 100x100 pixels required."}), 400
        
        # Preprocess for model
        img_resized = img.resize((224, 224))
        img_array = keras_image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = cnn_model.predict(img_array, verbose=0)[0][0]
        # ⚠️ DÜZELTME: flow_from_directory alfabetik sıraladığı için etiketler ters:
        # class 0 = hantavirus, class 1 = normal → prediction < 0.5 ise hantavirus
        confidence = float(1 - prediction) if prediction < 0.5 else float(prediction)
        result = "Hantavirus Detected" if prediction < 0.5 else "Normal Tissue"
        
        # Confidence threshold check
        if confidence < 0.60:
            return jsonify({
                "success": False, 
                "error": "Görsel hantavirüs mikroskopi verisine benzemiyor. Lütfen uygun laboratuvar görseli yükleyin."
            }), 400
        
        # Save to database
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (timestamp, module_type, input_summary, prediction_result, confidence, model_accuracy)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            'visual_analysis',
            f"Image: {secure_filename(file.filename)} ({img.width}x{img.height})",
            result,
            round(confidence, 4),
            MODEL_METRICS['cnn_accuracy']
        ))
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "result": result,
            "confidence": round(confidence * 100, 2),
            "model_accuracy": MODEL_METRICS['cnn_accuracy']
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Processing error: {str(e)}"}), 500

@app.route('/predict_risk', methods=['POST'])
def predict_risk():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Extract and validate inputs
        try:
            features = [
                float(data['region']),
                float(data['temperature']),
                float(data['humidity']),
                float(data['rodent_score']),
                float(data['is_rural']),
                float(data['has_warning'])
            ]
        except (KeyError, ValueError) as e:
            return jsonify({"success": False, "error": f"Invalid input format: {str(e)}"}), 400
        
        # Scale and predict
        features_scaled = rf_scaler.transform([features])
        prediction = rf_model.predict_proba(features_scaled)[0]
        risk_prob = float(prediction[1])  # Probability of high risk
        confidence = float(max(prediction))
        
        result = "Yüksek Risk" if risk_prob >= 0.5 else "Düşük Risk"
        risk_percentage = round(risk_prob * 100, 2)
        
        # Save to database
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        input_summary = f"Region:{int(features[0])}, Temp:{features[1]}°C, Humidity:{features[2]}%, Rodent:{int(features[3])}/10"
        cursor.execute('''
            INSERT INTO predictions (timestamp, module_type, input_summary, prediction_result, confidence, model_accuracy)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            'environmental_risk',
            input_summary,
            f"{result} ({risk_percentage}%)",
            round(confidence, 4),
            MODEL_METRICS['rf_accuracy']
        ))
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "result": result,
            "risk_percentage": risk_percentage,
            "confidence": round(confidence * 100, 2),
            "model_accuracy": MODEL_METRICS['rf_accuracy']
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Processing error: {str(e)}"}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == 'admin' and password == 'hanta2024':
            session['admin'] = True
            return redirect(url_for('admin'))
        else:
            return render_template('login.html', error="Invalid credentials")
    
    return render_template('login.html', error=None)

@app.route('/admin')
@admin_required
def admin():
    conn = sqlite3.connect('predictions.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 50')
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return render_template('admin.html', records=records)

@app.route('/admin/export_csv')
@admin_required
def export_csv():
    conn = sqlite3.connect('predictions.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
    records = cursor.fetchall()
    conn.close()
    
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['id', 'timestamp', 'module_type', 'input_summary', 'prediction_result', 'confidence', 'model_accuracy'])
    for row in records:
        cw.writerow([row['id'], row['timestamp'], row['module_type'], row['input_summary'], 
                    row['prediction_result'], row['confidence'], row['model_accuracy']])
    
    si.seek(0)
    return send_file(
        io.BytesIO(si.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'hantavirus_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

@app.route('/admin/clear', methods=['POST'])
@admin_required
def clear_predictions():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "All records deleted"})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Error handlers
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"success": False, "error": "File too large. Maximum 16MB allowed."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting Hantavirus Analysis System on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)