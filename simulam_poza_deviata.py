import cv2

cap = cv2.VideoCapture("C:\\Users\\Antonia\\Downloads\\V20251202_105058_001.avi")
ret, frame = cap.read()
cap.release()

# EXACT același ROI ca în analiză
y1, y2, x1, x2 = (299, 779, 666, 1313)
roi = frame[y1:y2, x1:x2]

cv2.imwrite("reference.jpg", roi)
