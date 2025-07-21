import cv2
import mediapipe as mp
import pyautogui
from math import hypot

# Init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

click_down = False
prev_x, prev_y = 0, 0
zoom_mode = 'none'
scrolling = False

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            lm = hand_landmarks.landmark
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Determine hand label: Left or Right
            hand_label = handedness.classification[0].label

            # Get coordinates of index and thumb tips
            x1 = int(lm[8].x * w)
            y1 = int(lm[8].y * h)
            x2 = int(lm[4].x * w)
            y2 = int(lm[4].y * h)

            # Move cursor with left hand only
            if hand_label == "Right":
                screen_x = int(lm[8].x * screen_w)
                screen_y = int(lm[8].y * screen_h)
                pyautogui.moveTo(screen_x, screen_y)

            # Draw circle on index finger tip
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

            # Distance between index and thumb for click
            distance = hypot(x2 - x1, y2 - y1)

            # Click with left hand
            if hand_label == "Right":
                if distance < 20:
                    if not click_down:
                        click_down = True
                        pyautogui.click()
                else:
                    click_down = False

                # Save current position
                prev_x, prev_y = x1, y1


    cv2.imshow("Hand Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()