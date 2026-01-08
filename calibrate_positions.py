import cv2
import numpy as np
from advanced_tire_qc import AdvancedTireQualityChecker, MM_TO_PX

# ================= CONFIG =================
RTSP_URL = "rtsp://user:pass@ip:port/stream"
MIN_AREA = 200        # ignorăm zgomot
FRAME_WAIT = 30       # frame-uri până la stabilizare
# ==========================================

checker = AdvancedTireQualityChecker()
checker.set_current_pattern("YAWG")  # TEMPORAR – doar pt color_ranges

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("❌ Nu pot deschide RTSP")

print("📡 Aștept stabilizare stream...")
for _ in range(FRAME_WAIT):
    cap.read()

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("❌ Nu pot citi frame")

display = frame.copy()
center_x = None

# ========== CLICK PE CENTRU ==========
def click(event, x, y, flags, param):
    global center_x
    if event == cv2.EVENT_LBUTTONDOWN:
        center_x = x
        disp = display.copy()
        cv2.line(disp, (x, 0), (x, disp.shape[0]), (0, 255, 0), 2)
        cv2.imshow("Calibrare centru banda", disp)

cv2.imshow("Calibrare centru banda", display)
cv2.setMouseCallback("Calibrare centru banda", click)

print("👉 Click pe CENTRUL benzii, apoi apasă orice tastă")
cv2.waitKey(0)
cv2.destroyAllWindows()

if center_x is None:
    raise RuntimeError("❌ Nu a fost selectat centrul")

print(f"\n✅ Centru bandă: x = {center_x}px ({center_x/MM_TO_PX:.1f} mm)\n")

# ========== AUTO-DETECT LINII ==========
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
stats = checker._calculate_image_statistics(frame)

results = {}

print("===== MASURARE LINII =====")

for color, ranges in checker.current_pattern.color_ranges.items():
    mask = checker._adaptive_color_detection(hsv, ranges, stats)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]

    if not contours:
        print(f"{color.upper():7} ❌ NU DETECTAT")
        continue

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cx = x + w // 2

    dist_px = abs(cx - center_x)
    dist_mm = dist_px / MM_TO_PX

    results[color] = round(dist_mm, 1)

    print(
        f"{color.upper():7} | "
        f"x={cx:4}px | "
        f"Δ={dist_px:4}px | "
        f"{dist_mm:6.1f} mm"
    )

# ========== OUTPUT FINAL ==========
print("\n===== COPY-PASTE ÎN PATTERN =====")
print("expected_positions_mm = {")
for c, mm in results.items():
    print(f'    "{c}": {int(round(mm))},')
print("}")
