import cv2
import mediapipe as mp
import numpy as np
import os
import time

# 설정
action_name = input("✋ 수집할 동작 이름을 입력하세요: ")  # 예: hello
SEQ_LENGTH = 30
TARGET_COUNT = 30

# 카메라 설정
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()
print("카메라 연결 성공")

# MediaPipe 손 모델 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# 데이터 수집 시작 알림
print("카메라 시작, 3초 후 수집 시작...")
time.sleep(3)
print(f"'{action_name}' 데이터 수집 시작!")

sequence = []
saved_count = 0

# 특징 추출 함수
def extract_features(landmarks):
    base = landmarks[0]
    features = []
    for i in range(1, len(landmarks)):
        dx = landmarks[i].x - base.x
        dy = landmarks[i].y - base.y
        dz = landmarks[i].z - base.z
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        features.append(dist)

    def angle(a, b, c):
        a, b, c = np.array([a.x, a.y, a.z]), np.array([b.x, b.y, b.z]), np.array([c.x, c.y, c.z])
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6) # 선형 방정식식
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) # 아크코사인


    angles = [
        angle(landmarks[4], landmarks[3], landmarks[2]),
        angle(landmarks[8], landmarks[7], landmarks[6]),
        angle(landmarks[12], landmarks[11], landmarks[10]),
        angle(landmarks[16], landmarks[15], landmarks[14]),
        angle(landmarks[20], landmarks[19], landmarks[18]),
    ]
    features.extend(angles)
    return features

# 수집 루프
with hands:
    while cap.isOpened() and saved_count < TARGET_COUNT:
        ret, frame = cap.read()
        if not ret:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            features = []

            # 감지된 손 2개까지 처리 (없으면 0으로 패딩)
            for i in range(2):
                if i < len(results.multi_hand_landmarks):
                    hand_landmarks = results.multi_hand_landmarks[i]
                    f = extract_features(hand_landmarks.landmark)
                    # 손 관절 시각화
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                else:
                    f = [0.0] * 25  # 손이 없으면 0으로 채움 (패딩)
                features.extend(f)  # 두 손 합쳐서 50차원

            sequence.append(features)

            # 프레임 저장
            raw_save_dir = f'data/raw/{action_name}/{int(time.time())}_{saved_count}'
            os.makedirs(raw_save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(raw_save_dir, f'{len(sequence)}.jpg'), frame)

            if len(sequence) == SEQ_LENGTH:
                # 시퀀스 저장
                seq_save_dir = f'data/seq/{action_name}'
                os.makedirs(seq_save_dir, exist_ok=True)
                npy_path = os.path.join(seq_save_dir, f'seq_{int(time.time())}.npy')
                np.save(npy_path, np.array(sequence))
                print(f"저장 완료: {npy_path}")
                saved_count += 1
                sequence = []

        else:
            sequence = []

        # 저장 진행률 표시
        cv2.putText(frame, f'{action_name} : {saved_count}/{TARGET_COUNT}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow('Collecting Data...', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 종료
cap.release()
cv2.destroyAllWindows()
print("수집 종료")
