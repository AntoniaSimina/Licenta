import cv2
import os

VIDEO_PATH = "C:\\Users\\Antonia\\Downloads\\V20251202_105058_001.avi"

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Nu pot citi video-ul")
    exit()

print("➡️ Selecteaza ROI cu mouse-ul si apasa ENTER sau SPACE")

roi = cv2.selectROI(
    "Selectare banda cauciuc",
    frame,
    fromCenter=False,
    showCrosshair=True
)

cv2.destroyAllWindows()

x, y, w, h = roi

print("\nROI SELECTAT:")
print(f"roi = ({y}, {y+h}, {x}, {x+w})")
