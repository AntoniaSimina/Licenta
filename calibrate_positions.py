import cv2
import numpy as np
from advanced_tire_qc import AdvancedTireQualityChecker, MM_TO_PX

# ================= CONFIG =================
# SOURCE: "local" sau "rtsp" (nu modifica logica mai jos)
SOURCE = "local"   # "local" | "rtsp"

# Calea catre fisierul local (folosit cand SOURCE == "local")
VIDEO_PATH = r"C:\Users\Antonia\Downloads\V20251202_105058_001.avi"

RTSP_URL = "rtsp://user:pass@ip:port/stream"
MIN_AREA = 200        # ignorăm zgomot
FRAME_WAIT = 30       # frame-uri până la stabilizare
# ==========================================

checker = AdvancedTireQualityChecker()
checker.set_current_pattern("YAWG")  # TEMPORAR – doar pt color_ranges

# Deschidem captura în funcție de SOURCE; restul logicii rămâne neschimbată
if SOURCE == "local":
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Nu pot deschide video local: {VIDEO_PATH}")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Nu pot citi frame din video local")

elif SOURCE == "rtsp":
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Nu pot deschide RTSP")

    print("Astept stabilizare stream...")
    for _ in range(FRAME_WAIT):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Nu pot citi frame")

else:
    raise ValueError("SOURCE trebuie sa fie 'local' sau 'rtsp'")

y1 = y2 = x1 = x2 = None
roi_frame = None
try:
    from calibrate_center_click import ROI
    y1, y2, x1, x2 = ROI
    roi_frame = frame[y1:y2, x1:x2]
    print(f"Using ROI from calibrate_center_click: {ROI}")
except Exception:
    try:
        import app as app_module
        ROI = getattr(app_module, 'ROI', None)
        if ROI is not None:
            y1, y2, x1, x2 = ROI
            roi_frame = frame[y1:y2, x1:x2]
            print(f"Using ROI from app module-level ROI: {ROI}")
        else:
            import ast, re
            with open('c:\\Users\\Antonia\\Desktop\\Licenta_2.0\\app.py', 'r', encoding='utf-8') as f:
                src = f.read()
            m = re.search(r"self\.roi\s*=\s*\(([^\)]+)\)", src)
            if m:
                tuple_text = '(' + m.group(1) + ')'
                ROI = ast.literal_eval(tuple_text)
                y1, y2, x1, x2 = ROI
                roi_frame = frame[y1:y2, x1:x2]
                print(f"Using ROI parsed from app.py: {ROI}")
    except Exception:
        roi_frame = frame.copy()
        x1 = 0
        y1 = 0

display = roi_frame.copy()
clicks = []

def click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    clicks.append((x, y))
    disp = display.copy()
    for i, (cx, cy) in enumerate(clicks):
        color = (0, 255, 0) if i == 0 else (0, 128, 255)
        cv2.drawMarker(disp, (cx, cy), color, cv2.MARKER_CROSS, 12, 2)
    cv2.imshow("Calibrare pozitii (ROI)", disp)


cv2.namedWindow("Calibrare pozitii (ROI)", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Calibrare pozitii (ROI)", display)
cv2.setMouseCallback("Calibrare pozitii (ROI)", click)

num_lines = len(checker.current_pattern.colors)
print("Click pe CENTRU benzii, apoi click pe fiecare linie in ORDINEA pattern (aqua, yellow, white, green).")
print(f"Total clicks asteptate: {1 + num_lines}")

while True:
    key = cv2.waitKey(10)
    if len(clicks) >= 1 + num_lines:
        break
    if key == 27:  # ESC cancels
        cv2.destroyAllWindows()
        raise RuntimeError("Calibrare anulată de utilizator")

cv2.destroyAllWindows()

center_click = clicks[0]
center_x_roi = center_click[0]
center_x_global = x1 + center_x_roi

print(f"\nCentru selectat (ROI): x={center_x_roi}px  -> GLOBAL x={center_x_global}px ({center_x_global/MM_TO_PX:.1f} mm)")

hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
stats = checker._calculate_image_statistics(roi_frame)
results = {}
print("===== MASURARE LINII (din click-uri) =====")

for i, color in enumerate(checker.current_pattern.colors):
    click_x, click_y = clicks[1 + i]
    global_x = x1 + click_x
    signed_px = int(click_x - center_x_roi)
    signed_mm = signed_px / MM_TO_PX

    results[color] = {
        "signed_mm": round(signed_mm, 1),
        "abs_mm": round(abs(signed_mm), 1),
        "global_x_px": int(global_x),
        "roi_x_px": int(click_x)
    }

    expected_mm = checker.current_pattern.expected_positions_mm.get(color)
    expected_px = checker.current_pattern.expected_positions_px.get(color)
    abs_err_mm = (abs(signed_mm) - expected_mm) if expected_mm is not None else None

    side = "R" if signed_px > 0 else ("L" if signed_px < 0 else "C")

    print(f"{color.upper():7} | roi_x={click_x:4}px | global_x={global_x:4}px | signedΔ={signed_px:4}px | {signed_mm:6.1f} mm ({side})"
          + (f" | expected={expected_mm}mm ({expected_px}px) | err={abs_err_mm:.1f}mm" if expected_mm is not None else ""))

    ranges = checker.current_pattern.color_ranges.get(color)
    mask = checker._adaptive_color_detection(hsv, ranges, stats)

    cv2.imwrite(f"debug_mask_{color}.png", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    print(f"{color.upper():7} — contururi totale: {len(contours)}, aria_max: {max(areas) if areas else 0}")

    contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]

    if not contours:
        print(f"{color.upper():7} NU DETECTAT (dupa filtrare MIN_AREA={MIN_AREA})")
        continue

    largest = max(contours, key=cv2.contourArea)

    M = cv2.moments(largest)
    if M.get("m00", 0) != 0:
        cx = int(M["m10"] / M["m00"]) 
    else:
        bx, by, bw, bh = cv2.boundingRect(largest)
        cx = bx + bw // 2

    dist_px_signed = int(cx - center_x_roi)
    dist_mm_signed = dist_px_signed / MM_TO_PX

    results.setdefault(color, {})
    results[color].update({
        "measured_cx_roi_px": int(cx),
        "measured_cx_global_px": int(x1 + cx),
        "measured_signed_mm": round(dist_mm_signed, 1),
        "measured_abs_mm": round(abs(dist_mm_signed), 1)
    })

    expected_mm = checker.current_pattern.expected_positions_mm.get(color)
    expected_px = checker.current_pattern.expected_positions_px.get(color)
    abs_err_mm = (abs(dist_mm_signed) - expected_mm) if expected_mm is not None else None

    side = "R" if dist_px_signed > 0 else ("L" if dist_px_signed < 0 else "C")

    print(
        f"{color.upper():7} | x={cx:4}px (roi) | signedΔ={dist_px_signed:4}px | {dist_mm_signed:6.1f} mm ({side})"
        + (f" | expected={expected_mm}mm ({expected_px}px) | err={abs_err_mm:.1f}mm" if expected_mm is not None else "")
    )

print("\n===== COPY-PASTE IN PATTERN =====")
print("expected_positions_mm = {")
for c, info in results.items():
    if isinstance(info, dict):
        mm_val = info.get("measured_abs_mm") or info.get("abs_mm") or 0
    else:
        mm_val = info
    print(f'    "{c}": {int(round(mm_val))},')
print("}")

print("\n===== POSITII SEMNATE (mm fata de centru) =====")
for c, info in results.items():
    if isinstance(info, dict):
        signed = info.get("measured_signed_mm") or info.get("signed_mm") or 0.0
    else:
        signed = 0.0
    print(f'    "{c}": {signed:+.1f} mm')
