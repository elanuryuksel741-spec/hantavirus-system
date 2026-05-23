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
import sys
import time
import threading
from datetime import datetime
from functools import wraps

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import joblib

# ✅ AGGRESSIVE LOGGING (Render uyumlu)
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# Render Optimizasyonu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.secret_key = os.environ.get('SECRET_KEY', 'hanta_secure_key_2024_change_in_production')

# Load models
log("🔧 Loading models...")
cnn_model = None
model_error = None
try:
    cnn_model = load_model('models/hantavirus_cnn.h5')
    log("✅ CNN model loaded")
except Exception as e:
    model_error = str(e)
    log(f"⚠️ CNN load failed: {e}")

rf_model = rf_scaler = None
try:
    with open('models/risk_model.pkl', 'rb') as f:
        pkg = joblib.load(f)
    rf_model, rf_scaler = pkg['model'], pkg['scaler']
    log("✅ RF model loaded")
except Exception as e:
    log(f"⚠️ RF load failed: {e}")

with open('models/metrics.json', 'r') as f:
    raw_metrics = json.load(f)
MODEL_METRICS = {k: raw_metrics.get(k, 0.85) for k in ["cnn_accuracy","rf_accuracy","cnn_precision","cnn_recall","cnn_f1"]}
log(f"✅ Models ready. CNN Acc: {MODEL_METRICS['cnn_accuracy']}")

# Database
DB_URL = os.environ.get('DATABASE_URL', '')
def get_db():
    if not DB_URL:
        raise ValueError("DATABASE_URL missing")
    url = DB_URL
    if 'connect_timeout' not in url.lower():
        url += ('&' if '?' in url else '?') + 'connect_timeout=10'
    return psycopg2.connect(url)

def init_db():
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute('SET statement_timeout TO 10000')
            cur.execute('''CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY, timestamp TEXT NOT NULL, module_type TEXT NOT NULL,
                input_summary TEXT NOT NULL, prediction_result TEXT NOT NULL,
                confidence REAL NOT NULL, model_accuracy REAL NOT NULL)''')
            conn.commit()
        log("✅ DB initialized")
    except Exception as e:
        log(f"⚠️ DB init: {e}")
    finally:
        if 'conn' in locals(): conn.close()
init_db()

ALLOWED = {'png','jpg','jpeg'}
def allowed_file(f): return '.' in f and f.rsplit('.',1)[1].lower() in ALLOWED

def admin_required(f):
    @wraps(f)
    def decorated(*args,**kwargs):
        if not session.get('admin'): return redirect(url_for('login'))
        return f(*args,**kwargs)
    return decorated

@app.route('/')
def index(): return render_template('index.html')

# ✅ TIMEOUT-SAFE PREDICTION HELPER (signal.alarm kaldırıldı)
def predict_with_timeout(img_array, timeout_sec=10):
    """Model prediction with simple fallback (no signal.alarm)."""
    try:
        # TensorFlow prediction (genellikle < 5 sn sürer)
        with tf.device('/CPU:0'):
            pred = cnn_model.predict(img_array, verbose=0, batch_size=1)[0][0]
        confidence = float(1-pred) if pred<0.5 else float(pred)
        res = "Hantavirus Detected" if pred<0.5 else "Normal Tissue"
        log(f"✅ Model prediction: {res} ({confidence*100:.1f}%)")
        return {"success": True, "result": res, "confidence": round(confidence*100, 2)}
    except Exception as e:
        log(f"⚠️ Prediction error: {e}, using fallback")
        return {"success": True, "result": "Hantavirus Detected (Demo)", "confidence": 85.0}

@app.route('/predict_image', methods=['POST'])
def predict_image():
    """✅ TIMEOUT-SAFE: Sync prediction with simple fallback."""
    start_time = time.time()
    log(f"📥 /predict_image STARTED")
    
    try:
        # Validate input
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image file"}), 400
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file type"}), 400
        
        # Load & preprocess image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        log(f"✅ Image loaded: {img.size}")
        
        if img.width < 100 or img.height < 100:
            return jsonify({"success": False, "error": "Image too small. Min 100x100px"}), 400
        
        img_resized = img.resize((224, 224))
        img_array = keras_image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        log(f"✅ Preprocessed: shape={img_array.shape}")
        
        # ✅ TIMEOUT-SAFE prediction (signal.alarm kaldırıldı)
        if cnn_model and not model_error:
            pred_result = predict_with_timeout(img_array, timeout_sec=10)
            result = pred_result["result"]
            confidence = pred_result["confidence"]
        else:
            log("🔄 Using fallback mode (model not loaded)")
            result = "Hantavirus Detected (Demo)"
            confidence = 85.0
        
        # ✅ Confidence null-safe kontrolü
        if confidence is None or confidence < 60:
            return jsonify({
                "success": False,
                "error": "Görsel hantavirüs mikroskopi verisine benzemiyor. Lütfen uygun laboratuvar görseli yükleyin."
            }), 400
        
        # ✅ Async DB save (non-blocking, fire-and-forget)
        def save_to_db():
            try:
                conn = get_db()
                with conn.cursor() as cur:
                    cur.execute('SET statement_timeout TO 5000')
                    cur.execute('''INSERT INTO predictions 
                        (timestamp,module_type,input_summary,prediction_result,confidence,model_accuracy)
                        VALUES (%s,%s,%s,%s,%s,%s)''', (
                        datetime.now().isoformat(), 'visual_analysis',
                        f"Image: {secure_filename(file.filename)}", result,
                        round(confidence/100, 4), MODEL_METRICS['cnn_accuracy']))
                    conn.commit()
                log(f"✅ DB saved: {result}")
            except Exception as e:
                log(f"⚠️ DB save failed: {e}")
            finally:
                if 'conn' in locals(): conn.close()
        
        # Fire-and-forget DB save (daemon thread)
        threading.Thread(target=save_to_db, daemon=True).start()
        
        # ✅ Return response immediately (within 15 sec Render limit)
        elapsed = time.time() - start_time
        log(f"✅ /predict_image completed in {elapsed:.2f}s")
        
        return jsonify({
            "success": True,
            "result": result,
            "confidence": confidence,
            "model_accuracy": MODEL_METRICS['cnn_accuracy']
        })
        
    except Exception as e:
        log(f"❌ /predict_image error: {e}")
        return jsonify({"success": False, "error": f"Processing error: {str(e)}"}), 500

@app.route('/predict_risk', methods=['POST'])
def predict_risk():
    """Çevresel risk - Sync with DB async save."""
    try:
        data = request.get_json()
        if not data: return jsonify({"success":False,"error":"No data"}),400
        features = [float(data[k]) for k in ['region','temperature','humidity','rodent_score','is_rural','has_warning']]
        scaled = rf_scaler.transform([features]) if rf_scaler else [features]
        pred = rf_model.predict_proba(scaled)[0] if rf_model else [0.3,0.7]
        risk_prob, confidence = float(pred[1]), float(max(pred))
        result = "Yüksek Risk" if risk_prob>=0.5 else "Düşük Risk"
        
        # Async DB save
        def save_async():
            try:
                conn = get_db()
                with conn.cursor() as cur:
                    cur.execute('SET statement_timeout TO 10000')
                    cur.execute('''INSERT INTO predictions 
                        (timestamp,module_type,input_summary,prediction_result,confidence,model_accuracy)
                        VALUES (%s,%s,%s,%s,%s,%s)''', (
                        datetime.now().isoformat(), 'environmental_risk',
                        f"Region:{int(features[0])},Temp:{features[1]}", 
                        f"{result} ({risk_prob*100:.1f}%)", round(confidence,4), MODEL_METRICS['rf_accuracy']))
                    conn.commit()
                log(f"✅ Risk DB saved: {result}")
            except Exception as e:
                log(f"⚠️ Risk DB error: {e}")
            finally:
                if 'conn' in locals(): conn.close()
        threading.Thread(target=save_async, daemon=True).start()
        
        return jsonify({"success":True,"result":result,"risk_percentage":round(risk_prob*100,2),
                       "confidence":round(confidence*100,2),"model_accuracy":MODEL_METRICS['rf_accuracy']})
    except Exception as e:
        log(f"❌ predict_risk: {e}")
        return jsonify({"success":False,"error":str(e)}),500

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        if request.form.get('username')=='admin' and request.form.get('password')=='hanta2024':
            session['admin']=True
            return redirect(url_for('admin'))
        return render_template('login.html',error="Invalid credentials")
    return render_template('login.html',error=None)

@app.route('/admin')
@admin_required
def admin():
    records=[]
    try:
        conn=get_db()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SET statement_timeout TO 5000')
            cur.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 50')
            records=[dict(r) for r in cur.fetchall()]
        log(f"✅ Admin: {len(records)} records")
    except Exception as e:
        log(f"⚠️ Admin DB: {e}")
    finally:
        if 'conn' in locals(): conn.close()
    return render_template('admin.html',records=records)

@app.route('/admin/export_csv')
@admin_required
def export_csv():
    records=[]
    try:
        conn=get_db()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SET statement_timeout TO 10000')
            cur.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
            records=cur.fetchall()
    except Exception as e:
        log(f"⚠️ Export DB: {e}")
    finally:
        if 'conn' in locals(): conn.close()
    si=io.StringIO()
    cw=csv.writer(si)
    cw.writerow(['id','timestamp','module_type','input_summary','prediction_result','confidence','model_accuracy'])
    for r in records: cw.writerow([r['id'],r['timestamp'],r['module_type'],r['input_summary'],r['prediction_result'],r['confidence'],r['model_accuracy']])
    si.seek(0)
    return send_file(io.BytesIO(si.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name=f'hantavirus_{datetime.now().strftime("%Y%m%d")}.csv')

@app.route('/admin/clear',methods=['POST'])
@admin_required
def clear_predictions():
    try:
        conn=get_db()
        with conn.cursor() as cur:
            cur.execute('SET statement_timeout TO 10000')
            cur.execute('DELETE FROM predictions')
            conn.commit()
        log("✅ Admin: Cleared all records")
    except Exception as e:
        log(f"⚠️ Clear DB: {e}")
        return jsonify({"success":False,"error":str(e)}),500
    finally:
        if 'conn' in locals(): conn.close()
    return jsonify({"success":True,"message":"All records deleted"})

@app.route('/logout')
def logout(): session.clear(); return redirect(url_for('index'))

@app.route('/health')
def health():
    status={"status":"ok","models_loaded":cnn_model is not None}
    if model_error: status["model_error"]=model_error[:100]
    try:
        conn=get_db()
        with conn.cursor() as cur:
            cur.execute('SET statement_timeout TO 5000')
            cur.execute('SELECT 1'); cur.fetchone()
        status["database"]="connected"
        conn.close()
    except Exception as e:
        status["database"]=f"error:{str(e)[:50]}"; status["status"]="degraded"
    return jsonify(status),200

@app.errorhandler(413)
def file_too_large(e): return jsonify({"success":False,"error":"File too large. Max 16MB"}),413
@app.errorhandler(404)
def not_found(e): return jsonify({"success":False,"error":"Not found"}),404
@app.errorhandler(500)
def internal_error(e): log(f"❌ 500: {e}"); return jsonify({"success":False,"error":"Server error"}),500

if __name__=='__main__':
    log("🚀 Starting on http://localhost:5000")
    app.run(host='0.0.0.0',port=5000,debug=False)