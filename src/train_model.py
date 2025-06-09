import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from collections import Counter

# 시퀀스 데이터가 저장된 디렉터리
DATA_DIR = "data/seq"

# 데이터를 담을 리스트 초기화
motion_data = []
labels = []
label_map = {}

# 데이터 디렉토리 존재 여부 확인
if not os.path.exists(DATA_DIR):
    print("시퀀스 데이터 폴더가 존재하지 않습니다. 먼저 시퀀스 데이터를 생성하세요.")
    exit()

# 시퀀스 데이터(.npy) 로드
for root, dirs, files in os.walk(DATA_DIR):  # 🔥 하위 폴더까지 탐색
    for file in files:
        if not file.endswith(".npy") or not file.startswith("seq_"):
            continue

        filepath = os.path.join(root, file)

        try:
            data = np.load(filepath)
            # 시퀀스 길이 검증 (60프레임)
            if len(data.shape) != 2 or data.shape[0] != 30:
                print(f"{file} 형식이 잘못되었습니다. 건너뜀.")
                continue
        except Exception as e:
            print(f"{file} 로딩 오류: {e}")
            continue

        # 🔑 폴더명을 라벨로 사용 (ex: "one", "bo", "gun")
        label_name = os.path.basename(root)

        # 새로운 라벨이면 맵에 추가
        if label_name not in label_map:
            label_map[label_name] = len(label_map)

        # 데이터와 라벨 추가
        motion_data.append(data.tolist())
        labels.append(label_map[label_name])

# 라벨 분포 출력
print("라벨 분포:", Counter(labels))

# 데이터가 없을 경우 종료
if len(motion_data) == 0:
    print("학습할 데이터가 없습니다.")
    exit()

# 입력 데이터의 시퀀스 길이와 특징 수 파악
max_seq_len = max(len(seq) for seq in motion_data)       # 시퀀스 길이 (예: 60)
feature_len = len(motion_data[0][0])                      # 특징 수 (예: 50)

# 리스트를 넘파이 배열로 변환
X_padded = np.array(motion_data, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

# 학습용 / 검증용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# 총 클래스 수 계산
num_classes = len(label_map)

# 🔧 LSTM 모델 구성
model = keras.Sequential([
    keras.layers.Masking(mask_value=0.0, input_shape=(max_seq_len, feature_len)),  # 0으로 채운 부분 무시
    keras.layers.LSTM(64),                         # LSTM 레이어
    keras.layers.Dense(64, activation="relu"),     # 완전 연결층
    keras.layers.Dense(num_classes, activation="softmax")  # 클래스 수만큼 출력
])

# 모델 컴파일
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 모델 학습 시작
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# 학습된 모델 구조와 가중치 저장
with open("hand_gesture_model.json", "w") as json_file:
    json_file.write(model.to_json())
model.save_weights("hand_gesture_weights.h5")

# 라벨 맵 저장 (나중에 예측 시 사용)
with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

print("모델 학습 완료!")
