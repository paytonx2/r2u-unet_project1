import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras import backend as K

app = Flask(__name__)
CORS(app) # อนุญาตให้ Vercel เรียกใช้งานได้

# ต้องใส่ฟังก์ชัน Loss ที่คุณเขียนไว้ด้วย ไม่งั้นจะโหลดโมเดลไม่ได้
def dice_coeff(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

# โหลดโมเดล (ใช้ compile=False เพื่อประหยัดทรัพยากรตอนโหลด)
model = tf.keras.models.load_model('model_cnn.h5', 
    custom_objects={'dice_coeff': dice_coeff, 'dice_loss': dice_loss, 'combined_loss': combined_loss},
    compile=False)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    
    # Preprocessing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_rgb, (128, 128))
    img_input = img_input.astype('float32') / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # Predict
    pred_mask = model.predict(img_input)[0]
    mask_binary = (pred_mask > 0.35).astype(np.uint8)
    pixel_count = int(np.sum(mask_binary))
    
    status = "MISSING" if pixel_count >= 15 else "GOOD"
    
    return jsonify({'status': status, 'pixel_count': pixel_count})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)