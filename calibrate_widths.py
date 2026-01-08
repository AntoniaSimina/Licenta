import cv2
import numpy as np

from advanced_tire_qc import AdvancedTireQualityChecker

# === CONFIG ===
# Sursă: "local" sau "rtsp" (exact ca în find_colors_HSV.py)
SOURCE = "local"  # "local" | "rtsp"

# Video local (când SOURCE == "local")
VIDEO_PATH = r"C:\Users\Antonia\Downloads\V20251202_105058_001.avi"

# URL RTSP (când SOURCE == "rtsp")
RTSP_URL = "rtsp://user:pass@ip:port/stream"

ROI = (299, 779, 666, 1313)  # (y1, y2, x1, x2)
PATTERN_NAME = "YAWG"

# Sampling behavior
SINGLE_FRAME_MODE = False     # True = calibrare dintr-un singur frame (recomandat pentru RTSP)
WARMUP_FRAMES = 15            # drop first frames (RTSP often starts unstable)
GRAB_MAX_TRIES = 120          # max reads to obtain a valid frame (single-frame mode)

MAX_SAMPLES = 120             # number of frames to sample (multi-frame mode)
FRAME_STRIDE = 5              # take every Nth frame (multi-frame mode)
MIN_HITS_PER_COLOR = 20       # minimum detections per color to trust stats (multi-frame mode)


def robust_stats(values: list[float]) -> dict:
    arr = np.array(values, dtype=np.float32)
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _open_capture() -> cv2.VideoCapture:
    if SOURCE == "local":
        cap = cv2.VideoCapture(VIDEO_PATH)
    elif SOURCE == "rtsp":
        # menținem aceeași abordare ca în find_colors_HSV.py
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    else:
        raise ValueError("SOURCE trebuie sa fie 'local' sau 'rtsp'")

    if not cap.isOpened():
        return cap

    # Best-effort: reduce buffering (nu e suportat pe toate backend-urile)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass

    return cap


def _grab_single_frame(cap: cv2.VideoCapture) -> np.ndarray:
    # Warmup reads
    for _ in range(max(0, int(WARMUP_FRAMES))):
        cap.read()

    last_good = None
    for _ in range(max(1, int(GRAB_MAX_TRIES))):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        if frame.size == 0:
            continue
        last_good = frame
        break

    if last_good is None:
        raise RuntimeError("Nu am putut citi un frame valid din sursă (RTSP/fișier).")

    return last_good


def _process_frame(checker: AdvancedTireQualityChecker, frame: np.ndarray, samples_by_color: dict[str, list[float]]) -> None:
    y1, y2, x1, x2 = ROI
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    image_stats = checker._calculate_image_statistics(roi)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    for i, color_name in enumerate(checker.current_pattern.colors):
        ranges = checker.current_pattern.color_ranges[color_name]
        mask = checker._adaptive_color_detection(hsv, ranges, image_stats)
        line_info = checker._detect_advanced_line(mask, color_name, i)
        if not line_info:
            continue

        x, y, w, h = line_info["bounding_box"]
        line_mask = mask[y:y + h, x:x + w]
        eff_w = checker._measure_effective_width(line_mask)
        if eff_w <= 0:
            eff_w = float(line_info["width"])

        samples_by_color[color_name].append(float(eff_w))


def main() -> None:
    cap = _open_capture()
    if not cap.isOpened():
        src = VIDEO_PATH if SOURCE == "local" else RTSP_URL
        raise RuntimeError(f"Nu pot deschide sursa ({SOURCE}): {src}")

    checker = AdvancedTireQualityChecker()
    checker.set_current_pattern(PATTERN_NAME)
    checker.debug_mode = False

    samples_by_color: dict[str, list[float]] = {c: [] for c in checker.current_pattern.colors}

    if SINGLE_FRAME_MODE:
        frame = _grab_single_frame(cap)
        _process_frame(checker, frame, samples_by_color)
    else:
        # Warmup reads for RTSP stability (safe for file sources too)
        for _ in range(max(0, int(WARMUP_FRAMES))):
            cap.read()

        frame_idx = 0
        sampled = 0

        while sampled < MAX_SAMPLES:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            if frame_idx % FRAME_STRIDE != 0:
                continue

            _process_frame(checker, frame, samples_by_color)
            sampled += 1

    cap.release()

    print("\n=== CALIBRARE LĂȚIMI (px) ===")
    if SOURCE == "local":
        print(f"Video: {VIDEO_PATH}")
    else:
        print(f"RTSP: {RTSP_URL}")
    print(f"ROI: {ROI}")
    print(f"Pattern: {PATTERN_NAME}\n")

    recommended_widths = []
    for color in checker.current_pattern.colors:
        vals = samples_by_color[color]
        min_hits = 1 if SINGLE_FRAME_MODE else MIN_HITS_PER_COLOR
        if len(vals) < min_hits:
            print(f"- {color.upper()}: prea puține detecții (n={len(vals)}).")
            recommended_widths.append(None)
            continue

        st = robust_stats(vals)
        print(
            f"- {color.upper()}: n={st['n']} median={st['median']:.1f}px "
            f"p10={st['p10']:.1f}px p90={st['p90']:.1f}px std={st['std']:.1f}px"
        )
        recommended_widths.append(int(round(st["median"])))

    if all(v is not None for v in recommended_widths):
        print("\nRECOMANDARE expected_widths (în ordinea pattern.colors):")
        print(f"colors = {checker.current_pattern.colors}")
        print(f"expected_widths = {recommended_widths}")

        # Suggest a tolerance based on spread (p10..p90)
        spreads = []
        for color in checker.current_pattern.colors:
            vals = samples_by_color[color]
            st = robust_stats(vals)
            spread = max(1.0, st["p90"] - st["p10"])
            spreads.append(spread)
        avg_spread = float(np.mean(spreads))
        avg_med = float(np.mean([v for v in recommended_widths if v is not None]))
        suggested_tol = min(max((avg_spread / max(avg_med, 1.0)) * 0.6, 0.10), 0.40)

        print("\nRECOMANDARE tolerance_width (aproximativ):")
        print(f"tolerance_width ≈ {suggested_tol:.2f}  (ajustezi după cât de strict vrei)")
    else:
        print("\nNu pot recomanda expected_widths complet (lipsesc culori).")
        print("Încearcă să crești MAX_SAMPLES sau să micșorezi FRAME_STRIDE.")


if __name__ == "__main__":
    main()
