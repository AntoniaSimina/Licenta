import cv2

# === CONFIG ===
VIDEO_PATH = r"C:\Users\Antonia\Downloads\V20251202_105058_001.avi"
FRAME_INDEX = 50        # frame bun din video
ROI = (299, 779, 666, 1313)  # (y1, y2, x1, x2)

# =================

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Nu pot deschide video")

cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Nu pot citi frame-ul")

# aplicăm ROI
y1, y2, x1, x2 = ROI
roi = frame[y1:y2, x1:x2]
display = roi.copy()

center_x_roi = None

def click(event, x, y, flags, param):
    global center_x_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        center_x_roi = x
        disp = display.copy()
        cv2.line(disp, (x, 0), (x, disp.shape[0]), (0, 255, 0), 2)
        cv2.imshow("Calibrare centru banda (ROI)", disp)

cv2.imshow("Calibrare centru banda (ROI)", display)
cv2.setMouseCallback("Calibrare centru banda (ROI)", click)

print("👉 Click pe centrul benzii, apoi apasă orice tastă...")
cv2.waitKey(0)
cv2.destroyAllWindows()

if center_x_roi is None:
    raise RuntimeError("Nu a fost selectat niciun punct")

# coordonată globală
center_x_global = x1 + center_x_roi

print("\n===== REZULTAT CALIBRARE =====")
print(f"Centru ROI     : x = {center_x_roi}px")
print(f"Centru GLOBAL  : x = {center_x_global}px")
print("==============================")
