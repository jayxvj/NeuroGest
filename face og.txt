import cv2
from fer import FER
from datetime import datetime
import csv

# Initialize webcam and FER detector
cap = cv2.VideoCapture(0)
detector = FER(mtcnn=True)

# Create CSV file to log emotions
csv_file = open('emotion_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Emotion', 'Confidence'])

print("[INFO] Logging emotions... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions
    results = detector.detect_emotions(frame)

    for result in results:
        (x, y, w, h) = result["box"]
        emotion, score = max(result["emotions"].items(), key=lambda x: x[1])

        # Draw face box and emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion} ({int(score * 100)}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([timestamp, emotion, round(score, 3)])

    # Show webcam window
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
csv_file.close()
cv2.destroyAllWindows()
print("[INFO] Log saved to emotion_log.csv")