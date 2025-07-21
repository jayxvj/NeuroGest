import cv2
import pandas as pd
import os

from mediapipe.python.solutions.pose import Pose
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS

# CSV output file
CSV_FILE = 'pose_data.csv'

# Initialize MediaPipe pose detection
pose = Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

# If file doesn't exist, create it with headers
if not os.path.exists(CSV_FILE):
    columns = [f"{coord}{i}" for i in range(33) for coord in ['x', 'y', 'z']] + ["label"]
    pd.DataFrame(columns=columns).to_csv(CSV_FILE, index=False)

# Ask user for label
label = input("Enter label for this activity (e.g., walking, dancing): ").strip()

# Start webcam
cap = cv2.VideoCapture(0)
sample_count = 0

print("Press 's' to save a sample. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip + convert BGR to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process pose
    results = pose.process(rgb)

    # Draw landmarks + collect data
    if results.pose_landmarks:
        draw_landmarks(frame, results.pose_landmarks, POSE_CONNECTIONS)

        # Extract 33 (x, y, z) values
        landmarks = results.pose_landmarks.landmark
        row = []
        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z])

        # Wait for user input to save
        key = cv2.waitKey(1)
        if key == ord('s'):
            row.append(label)
            df = pd.read_csv(CSV_FILE)
            df.loc[len(df)] = row
            df.to_csv(CSV_FILE, index=False)
            sample_count += 1
            print(f"✅ Sample {sample_count} saved for '{label}'.")

        elif key == ord('q'):
            break

    # Show webcam feed
    cv2.imshow("Pose Data Collection", frame)

cap.release()
cv2.destroyAllWindows()
print(f"\n📦 Total samples collected for '{label}': {sample_count}")