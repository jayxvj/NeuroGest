import cv2
import mediapipe as mp
import joblib
import numpy as np
import warnings

# 🔇 Suppress protobuf UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load model and initialize MediaPipe
model = joblib.load("hand_sign_model.pkl")
expected_features = model.n_features_in_

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)  # mirror image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lmList = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

            # Dynamically extract x,y or x,y,z depending on model
            flat_landmarks = []
            for lm in lmList:
                if expected_features == 42:
                    flat_landmarks.extend([lm[0], lm[1]])        # x, y
                elif expected_features == 63:
                    flat_landmarks.extend([lm[0], lm[1], lm[2]]) # x, y, z

            if len(flat_landmarks) == expected_features:
                prediction = model.predict([flat_landmarks])[0]

                # Draw results
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x = int(lmList[0][0] * frame.shape[1])
                y = int(lmList[0][1] * frame.shape[0])
                cv2.putText(frame, prediction, (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Invalid landmark size", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()