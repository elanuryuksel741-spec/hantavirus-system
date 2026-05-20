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
import threading
from datetime import datetime
from functools import wraps

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import joblib

# Render Free Tier RAM/CPU Optimizasyonu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'hanta_secure_key_2024_change_in_production')

# Load models and metrics
print("🔧 Loading models...")
cnn_model = load_model('models/hantavirus_cnn.h5')
with open('models/risk_model.pkl', 'rb') as f:
    rf_package = joblib.load(f)
rf_model = rf_package['model']
rf_scaler = rf_package['scaler']
with open('models/metrics.json', 'r') as f:
    raw_metrics = json.load(f)

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

# Database setup (PostgreSQL - Neon.tech with pooler compatibility)
DB_URL = os.environ.get('DATABASE_URL', '')

def get_db():
    """PostgreSQL bağlantısı oluşturur, timeout parametrelerini connection string'e ekler."""
    if not DB_URL:
        raise ValueError("DATABASE_URL ortam değişkeni tanımlı değil!")
    
    # Timeout parametrelerini connection string'e ekle
    db_url = DB_URL
    if 'connect_timeout' not in db_url.lower():
        separator = '&' if '?' in db_url else '?'
        db_url = f"{db_url}{separator}connect_timeout=10"
    if 'statement_timeout' not in db_url.lower():
        separator = '&' if '?' in db_url else '?'
        # options parametresini doğru formatta ekle
        db_url = f"{db_url}{separator}options='-c statement_timeout=10000'"
    
    # Bağlantıyı kur
    conn = psycopg2.connect(db_url)
    return conn

def init_db():
    """Veritabanı tablosunu oluşturur (idempotent)."""
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    module_type TEXT NOT NULL,
                    input_summary TEXT NOT NULL,
                    prediction_result TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    model_accuracy REAL NOT NULL
                )
            ''')
            conn.commit()
    except psycopg2.OperationalError as e:
        print(f"⚠️ DB init warning (non-critical): {e}")
    except Exception as e:
        print(f"⚠️ DB init error: {e}")
    finally:
        if 'conn' in locals() and conn: conn.close()

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
    """Görsel analiz endpoint'i - Async DB yazma ile Render timeout önleme."""
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file type. Only JPG, JPEG, PNG allowed."}), 400
        
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        if img.width < 100 or img.height < 100:
            return jsonify({"success": False, "error": "Image too small. Minimum 100x100 pixels required."}), 400
        
        img_resized = img.resize((224, 224))
        img_array = keras_image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = cnn_model.predict(img_array, verbose=0)[0][0]
        confidence = float(1 - prediction) if prediction < 0.5 else float(prediction)
        result = "Hantavirus Detected" if prediction < 0.5 else "Normal Tissue"
        
        if confidence < 0.60:
            return jsonify({
                "success": False, 
                "error": "Görsel hantavirüs mikroskopi verisine benzemiyor. Lütfen uygun laboratuvar görseli yükleyin."
            }), 400
        
        # ✅ OPTIMIZASYON: Önce kullanıcıya yanıt döndür, DB yazma işlemini arka planda yap
        response_data = {
            "success": True,
            "result": result,
            "confidence": round(confidence * 100, 2),
            "model_accuracy": MODEL_METRICS['cnn_accuracy']
        }
        
        def save_to_db_async():
            """DB INSERT işlemini arka planda yapar, kullanıcı beklemez."""
            conn = None
            try:
                conn = get_db()
                with conn.cursor() as cur:
                    cur.execute('''
                        INSERT INTO predictions (timestamp, module_type, input_summary, prediction_result, confidence, model_accuracy)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    ''', (
                        datetime.now().isoformat(),
                        'visual_analysis',
                        f"Image: {secure_filename(file.filename)} ({img.width}x{img.height})",
                        result,
                        round(confidence, 4),
                        MODEL_METRICS['cnn_accuracy']
                    ))
                    conn.commit()
                print(f"✅ DB save success: {result}")
            except psycopg2.OperationalError as e:
                print(f"⚠️ DB write failed (pooler/timeout): {e}")
            except psycopg2.Error as e:
                print(f"⚠️ DB write error: {e}")
            except Exception as e:
                print(f"⚠️ Unexpected DB error: {e}")
            finally:
                if conn: conn.close()
        
        # Thread başlat (daemon=True: app kapanınca thread de kapanır)
        threading.Thread(target=save_to_db_async, daemon=True).start()
        
        # Hemen kullanıcıya yanıt döndür (timeout önleme)
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ predict_image error: {e}")
        return jsonify({"success": False, "error": f"Processing error: {str(e)}"}), 500

@app.route('/predict_risk', methods=['POST'])
def predict_risk():
    """Çevresel risk analiz endpoint'i - Async DB yazma ile Render timeout önleme."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
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
        
        features_scaled = rf_scaler.transform([features])
        prediction = rf_model.predict_proba(features_scaled)[0]
        risk_prob = float(prediction[1])
        confidence = float(max(prediction))
        
        result = "Yüksek Risk" if risk_prob >= 0.5 else "Düşük Risk"
        risk_percentage = round(risk_prob * 100, 2)
        
        # ✅ OPTIMIZASYON: Önce yanıt döndür, DB'yi async yaz
        response_data = {
            "success": True,
            "result": result,
            "risk_percentage": risk_percentage,
            "confidence": round(confidence * 100, 2),
            "model_accuracy": MODEL_METRICS['rf_accuracy']
        }
        
        def save_to_db_async():
            """DB INSERT işlemini arka planda yapar, kullanıcı beklemez."""
            conn = None
            try:
                conn = get_db()
                with conn.cursor() as cur:
                    input_summary = f"Region:{int(features[0])}, Temp:{features[1]}°C, Humidity:{features[2]}%, Rodent:{int(features[3])}/10"
                    cur.execute('''
                        INSERT INTO predictions (timestamp, module_type, input_summary, prediction_result, confidence, model_accuracy)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    ''', (
                        datetime.now().isoformat(),
                        'environmental_risk',
                        input_summary,
                        f"{result} ({risk_percentage}%)",
                        round(confidence, 4),
                        MODEL_METRICS['rf_accuracy']
                    ))
                    conn.commit()
                print(f"✅ DB save success: {result}")
            except psycopg2.OperationalError as e:
                print(f"⚠️ DB write failed (pooler/timeout): {e}")
            except psycopg2.Error as e:
                print(f"⚠️ DB write error: {e}")
            except Exception as e:
                print(f"⚠️ Unexpected DB error: {e}")
            finally:
                if conn: conn.close()
        
        threading.Thread(target=save_to_db_async, daemon=True).start()
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ predict_risk error: {e}")
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
    """Admin paneli - PostgreSQL'den kayıtları okur, timeout fallback ile."""
    records = []
    conn = None
    try:
        conn = get_db()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # ✅ Timeout'u SQL komutu ile ayarla (5 saniye max sorgu süresi)
            cur.execute('SET statement_timeout TO 5000')
            cur.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 50')
            records = [dict(row) for row in cur.fetchall()]
        print(f"✅ Admin: {len(records)} records loaded")
    except psycopg2.OperationalError as e:
        print(f"⚠️ Admin DB timeout (showing empty): {e}")
        records = []
    except psycopg2.Error as e:
        print(f"⚠️ Admin DB error: {e}")
        records = []
    except Exception as e:
        print(f"⚠️ Unexpected admin error: {e}")
        records = []
    finally:
        if conn: conn.close()
    
    return render_template('admin.html', records=records)

@app.route('/admin/export_csv')
@admin_required
def export_csv():
    """CSV export - PostgreSQL'den tüm kayıtları okur."""
    records = []
    conn = None
    try:
        conn = get_db()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
            records = cur.fetchall()
    except psycopg2.OperationalError as e:
        print(f"⚠️ Export DB read failed: {e}")
    except psycopg2.Error as e:
        print(f"⚠️ Export DB error: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected export error: {e}")
    finally:
        if conn: conn.close()
    
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
    """Tüm kayıtları sil - PostgreSQL."""
    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute('DELETE FROM predictions')
            conn.commit()
    except psycopg2.OperationalError as e:
        print(f"⚠️ Clear DB failed: {e}")
        return jsonify({"success": False, "error": "Database operation failed"}), 500
    except psycopg2.Error as e:
        print(f"⚠️ Clear DB error: {e}")
        return jsonify({"success": False, "error": "Database error"}), 500
    except Exception as e:
        print(f"⚠️ Unexpected clear error: {e}")
        return jsonify({"success": False, "error": "Internal error"}), 500
    finally:
        if conn: conn.close()
    
    return jsonify({"success": True, "message": "All records deleted"})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# 🆕 Health Check Endpoint (Render/Debug için)
@app.route('/health')
def health_check():
    """Render health check endpoint - DB bağlantısını test eder."""
    status = {"status": "ok", "models_loaded": True}
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute('SELECT 1')
            cur.fetchone()
        status["database"] = "connected"
        conn.close()
    except Exception as e:
        status["database"] = f"error: {str(e)[:50]}"
        status["status"] = "degraded"
    return jsonify(status), 200

# Error handlers
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"success": False, "error": "File too large. Maximum 16MB allowed."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    print(f"❌ Internal error: {e}")
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting Hantavirus Analysis System on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)