import cv2
import numpy as np

# Config
VIDEO_PATH = r"C:\Users\Lenovo\Downloads\files\V20260219_133420_001.avi"
HOMOGRAPHY_FILE = "matrice_omografie.npy"
WARPED_SIZE = (2750, 2000)
WINDOW_NAME = "Reglaj"

# Pattern test curent (modifica usor aici)
COTE_PRODUCTIE_MM = {
    "white": 45,
    "aqua": 55,
    "red": 65,
}


def nothing(_):
    pass


def find_dynamic_center_x(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    y_margin = int(height * 0.3)
    roi = gray[y_margin:height - y_margin, :]
    if roi.size == 0:
        return width // 2

    profile = np.mean(roi, axis=0)
    kernel = max(5, width // 50)
    if kernel % 2 == 0:
        kernel += 1

    smooth = cv2.GaussianBlur(
        profile.reshape(1, -1).astype(np.float32),
        (kernel, 1),
        0,
    ).flatten()

    margin = width // 10
    search = smooth[margin:width - margin]
    if search.size == 0:
        return width // 2

    return int(np.argmin(search) + margin)


def main():
    # 1. Incarca omografia
    M_homography = np.load(HOMOGRAPHY_FILE)

    # 2. Fereastra + trackbars
    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar("Scale_Adj", WINDOW_NAME, 100, 200, nothing)  # 100 => 1.0
    cv2.createTrackbar("Offset", WINDOW_NAME, 500, 1000, nothing)    # 500 => 0

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Nu pot deschide video-ul: {VIDEO_PATH}")

    last_center = WARPED_SIZE[0] // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
            continue

        # Pas A: Warp (imagine indreptata)
        frame_warped = cv2.warpPerspective(frame, M_homography, WARPED_SIZE)

        # Pas B: Citeste trackbars
        s_adj = cv2.getTrackbarPos("Scale_Adj", WINDOW_NAME) / 100.0
        off_adj = cv2.getTrackbarPos("Offset", WINDOW_NAME) - 500
        scale_final = 10.0 * s_adj

        # Pas C: Centru dinamic pe cadrul warped + stabilizare simpla
        detected_center = find_dynamic_center_x(frame_warped)
        centru_px = int(0.8 * last_center + 0.2 * detected_center)
        last_center = centru_px
        referinta_finala = centru_px + off_adj

        # Deseneaza referinta
        cv2.line(frame_warped, (centru_px, 0), (centru_px, WARPED_SIZE[1]), (0, 255, 255), 2)
        cv2.line(frame_warped, (referinta_finala, 0), (referinta_finala, WARPED_SIZE[1]), (255, 255, 0), 2)

        cv2.putText(
            frame_warped,
            f"Scale_Adj={s_adj:.3f}  SCALE_FINAL={scale_final:.3f}px/mm  Offset={off_adj}px",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame_warped,
            f"Centru dinamic={centru_px}px  Referinta={referinta_finala}px",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Linii teoretice
        for nume, cota_mm in COTE_PRODUCTIE_MM.items():
            # Pozitive la stanga centrului (conventia proiectului)
            pos_x = int(referinta_finala - (cota_mm * scale_final))
            pos_x = max(0, min(WARPED_SIZE[0] - 1, pos_x))

            cv2.line(frame_warped, (pos_x, 0), (pos_x, WARPED_SIZE[1]), (0, 255, 0), 2)
            cv2.putText(
                frame_warped,
                f"{nume} ({cota_mm}mm)",
                (max(5, pos_x + 5), 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        preview = cv2.resize(frame_warped, (960, 700))
        cv2.imshow(WINDOW_NAME, preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            print(
                f"VALORI FINALE: Scale_Adj={s_adj:.4f}, Offset={off_adj}, "
                f"SCALE_FINAL={scale_final:.4f}"
            )
            break
        if key in (ord("q"), ord("x"), ord("X")):
            break

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
