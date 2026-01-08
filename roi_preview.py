import cv2
import os

"""Previzualizare ROI dintr-un frame: local video sau RTSP."""

# Configurare sursă: "local" sau "rtsp"
SOURCE = "local"  # "local" | "rtsp"

# Calea video local (folosită când SOURCE == "local")
VIDEO_PATH = r"C:\\Users\\Antonia\\Downloads\\V20251202_105058_001.avi"

# URL RTSP (folosit când SOURCE == "rtsp")
RTSP_URL = "rtsp://user:pass@ip:port/stream"


def grab_first_frame():
    if SOURCE == "local":
        cap = cv2.VideoCapture(VIDEO_PATH)
    elif SOURCE == "rtsp":
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    else:
        raise ValueError("SOURCE trebuie sa fie 'local' sau 'rtsp'")

    if not cap.isOpened():
        print(f"❌ Nu pot deschide sursa ({SOURCE})")
        return None

    # Încearcă să citești un singur frame
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("❌ Nu am putut citi primul frame")
        return None

    return frame


def main():
    frame = grab_first_frame()
    if frame is None:
        return

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


if __name__ == "__main__":
    main()
