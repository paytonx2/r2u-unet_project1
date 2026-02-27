import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras import backend as K

app = Flask(__name__)
CORS(app) # สำคัญมาก: เพื่อให้ Vercel เรียกข้ามโดเมนได้

# --- Custom Functions สำหรับโมเดล R2U-Net ---
def dice_coeff(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

# --- Load Model ---
MODEL_PATH = 'defect_medel.h5'
model = tf.keras.models.load_model(MODEL_PATH, 
    custom_objects={'dice_coeff': dice_coeff, 'dice_loss': dice_loss, 'combined_loss': combined_loss},
    compile=False)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        conf_threshold = float(request.form.get('conf_threshold', 0.35))
        px_threshold = int(request.form.get('px_threshold', 15))

        # อ่านไฟล์ภาพ
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        h_orig, w_orig = img.shape[:2]

        # Preprocessing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img_rgb, (128, 128))
        img_input = img_input.astype('float32') / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        # Inference
        pred_mask = model.predict(img_input, verbose=0)[0]
        mask_binary = (pred_mask > conf_threshold).astype(np.uint8)
        
        # Resize mask กลับมาขนาดจริงเพื่อตรวจนับพิกเซล
        mask_full = cv2.resize(mask_binary, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        pixel_count = int(np.sum(mask_full))
        
        status = "MISSING" if pixel_count >= px_threshold else "GOOD"
        
        return jsonify({
            'status': status,
            'pixel_count': pixel_count,
            'success': True
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Render จะกำหนด PORT ให้เองผ่าน Environment Variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)