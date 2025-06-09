import eventlet
eventlet.monkey_patch()  # ✅ 이 줄을 반드시 가장 먼저!

import json
import numpy as np
from flask import Flask, request
from flask_socketio import SocketIO, emit
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# Flask & SocketIO 초기화
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# 학습된 모델 로드 함수
def load_model_and_labels():
    with open("hand_gesture_model.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("hand_gesture_weights.h5")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    with open("label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}
    return model, idx_to_label

model, idx_to_label = load_model_and_labels()
SEQ_LENGTH = 30
FEATURE_DIM = 50  # 팔 포함하여 54차원으로 변경

@socketio.on('predict_sequence')
def handle_predict_sequence(data):
    try:
        seq = np.array(data, dtype=np.float32)
        if seq.shape != (SEQ_LENGTH, FEATURE_DIM):
            emit('prediction_result', {
                'error': f'시퀀스 크기 오류: {seq.shape} (기대: {(SEQ_LENGTH, FEATURE_DIM)})'
            }, to=request.sid)
            return

        seq = np.expand_dims(seq, axis=0)  # (1, 30, 54)
        preds = model.predict(seq)
        pred_idx = np.argmax(preds)
        pred_label = idx_to_label.get(pred_idx, "Unknown")
        confidence = float(preds[0][pred_idx])

        emit('prediction_result', {
            'label': pred_label,
            'confidence': confidence
        }, to=request.sid)
    except Exception as e:
        emit('prediction_result', {'error': str(e)}, to=request.sid)

@app.route('/')
def index():
    return "손동작 인식 서버가 실행 중입니다."

if __name__ == '__main__':
    print("✅ 서버 실행 중: http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000)
