import cv2

"""Inspectează valori HSV prin hover într-un ROI, din video local sau RTSP."""

# Sursă: "local" sau "rtsp"
SOURCE = "local"  # "local" | "rtsp"

# Video local (când SOURCE == "local")
VIDEO_PATH = r"C:\\Users\\Antonia\\Downloads\\V20251202_105058_001.avi"

# URL RTSP (când SOURCE == "rtsp")
RTSP_URL = "rtsp://user:pass@ip:port/stream"

# ROI predefinit (y1, y2, x1, x2)
roi = (299, 779, 666, 1313)


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

    y1, y2, x1, x2 = roi
    roi_img = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            h, s, v = hsv[y, x]
            print(f"H={h}, S={s}, V={v}")

    cv2.imshow("ROI HSV DEBUG", roi_img)
    cv2.setMouseCallback("ROI HSV DEBUG", on_mouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
