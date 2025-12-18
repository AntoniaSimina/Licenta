import cv2

img = cv2.imread("reference.jpg")

# coordonate reale (le citești o singură dată din detect_reference_frame)
x_left = 200
x_right = 230
shift = 5

band = img[:, x_left:x_right].copy()

# mutăm banda
img[:, x_left+shift:x_right+shift] = band

# NU ștergem complet zona veche
# doar o estompăm ușor
img[:, x_left:x_left+shift] = img[:, x_left+shift:x_left+2*shift]

cv2.imwrite("test_shifted_small.jpg", img)
