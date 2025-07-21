import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import pandas as pd
from joblib import load
from mediapipe.python.solutions.pose import Pose
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS

# Load trained model
model = load("activity_model.pkl")

# Setup MediaPipe Pose
pose = Pose(static_image_mode=False, model_complexity=1)

# Webcam
cap = cv2.VideoCapture(0)

print("🔍 Predicting... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    activity = "Detecting..."

    if results.pose_landmarks:
        draw_landmarks(frame, results.pose_landmarks, POSE_CONNECTIONS)

        # Extract and flatten landmarks
        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])

        if len(row) == 99:  # 33 landmarks * 3 coords
            columns = [f"{coord}{i}" for i in range(33) for coord in ['x', 'y', 'z']]
            X = pd.DataFrame([row], columns=columns)
            activity = model.predict(X)[0]

    # Show activity label
    cv2.putText(frame, f'Activity: {activity}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Real-Time Activity Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()