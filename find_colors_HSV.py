import cv2

VIDEO_PATH = r"C:\\Users\\Antonia\\Downloads\\V20251202_105058_001.avi"
roi = (299, 779, 666, 1313)

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

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
