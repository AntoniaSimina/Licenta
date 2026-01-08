import cv2
import numpy as np

# ================= CONFIG =================
RTSP_URL = "rtsp://user:pass@ip:port/stream"
ROI = (299, 779, 666, 1313)  # (y1, y2, x1, x2)
MM_TO_PX = 3.2
SAVE_PREFIX = "panic_debug"
# ==========================================

print("🔴 PANIC BUTTON – pornire")

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("❌ Nu pot deschide RTSP")

ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    raise RuntimeError("❌ Nu pot citi frame")

print("✅ Frame capturat")

# ROI
y1, y2, x1, x2 = ROI
roi = frame[y1:y2, x1:x2].copy()

display = roi.copy()
center_x = None

# ================= CLICK CENTRU =================
def click(event, x, y, flags, param):
    global center_x
    if event == cv2.EVENT_LBUTTONDOWN:
        center_x = x
        tmp = display.copy()
        cv2.line(tmp, (x, 0), (x, tmp.shape[0]), (0, 255, 0), 2)
        cv2.imshow("PANIC BUTTON", tmp)

cv2.imshow("PANIC BUTTON", display)
cv2.setMouseCallback("PANIC BUTTON", click)

print("👉 Click pe CENTRUL benzii, apoi apasă orice tastă")
cv2.waitKey(0)
cv2.destroyAllWindows()

if center_x is None:
    raise RuntimeError("❌ Centru neales")

center_x_global = x1 + center_x
print(f"🟢 Centru ROI: {center_x}px | Global: {center_x_global}px")

# ================= HSV =================
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

COLOR_RANGES = {
    "green":  [([65, 25, 130], [90, 255, 255])],
    "white":  [([0, 0, 170], [180, 20, 255])],
    "yellow": [([18, 20, 140], [42, 255, 255])],
    "aqua":   [([90, 80, 160], [110, 255, 255])]
}

print("\n📏 POZIȚII DETECTATE:")

for color, ranges in COLOR_RANGES.items():
    mask = None
    for low, high in ranges:
        m = cv2.inRange(hsv, np.array(low), np.array(high))
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    cv2.imwrite(f"{SAVE_PREFIX}_mask_{color}.png", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 150]

    if not contours:
        print(f"❌ {color.upper()} – NU DETECTAT")
        continue

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cx = x + w // 2

    dist_px = abs(cx - center_x)
    dist_mm = dist_px / MM_TO_PX

    print(f"✅ {color.upper():6s} | x={cx:4d}px | dist={dist_mm:5.1f} mm")

    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.circle(display, (cx, display.shape[0] // 2), 5, (255, 0, 0), -1)

# ================= SAVE =================
cv2.line(display, (center_x, 0), (center_x, display.shape[0]), (0, 255, 0), 2)
cv2.imwrite(f"{SAVE_PREFIX}_final.png", display)

print("\n💾 Debug salvat:")
print(f" - {SAVE_PREFIX}_final.png")
print(f" - {SAVE_PREFIX}_mask_*.png")

cv2.imshow("PANIC RESULT", display)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("🧯 PANIC BUTTON – gata")
