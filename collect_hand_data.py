import os
import sys
import ctypes

# Set TF + absl suppression
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Hide native stderr (for TensorFlow Lite, absl, etc.)
def suppress_native_stderr():
    if os.name == 'nt':
        # Windows
        kernel32 = ctypes.windll.kernel32
        kernel32.SetStdHandle(-12, 0)  # -12 = STDERR
    else:
        # Unix/Linux
        sys.stderr = open(os.devnull, 'w')

suppress_native_stderr()

import cv2
import mediapipe as mp
import pandas as pd

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Setup webcam
cap = cv2.VideoCapture(0)

# Data storage
data = []
label = input("Enter the sign label (e.g., Hello, A, ThankYou): ")

print("[INFO] Press 's' to save landmark, 'q' to quit")

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Flatten landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Wait for keypress to save
            key = cv2.waitKey(1)
            if key == ord('s'):
                data.append(landmarks + [label])
                print(f"[SAVED] Sample saved for label: {label}")

    cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Data Collector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("hand_signs.csv", mode='a', index=False, header=False)
print(f"[INFO] Data saved to hand_signs.csv with label: {label}")