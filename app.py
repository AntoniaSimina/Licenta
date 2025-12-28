import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from advanced_tire_qc import AdvancedTireQualityChecker
import colorsys

VIDEO_PATH = r"C:\Users\Antonia\Desktop\Licenta_2.0\output_overlay.avi"

def hsv_to_bgr(h, s, v):
    h_norm = h / 180.0
    s_norm = s / 255.0
    v_norm = v / 255.0
    
    r, g, b = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)
    return (int(b * 255), int(g * 255), int(r * 255))

def generate_pattern_image(pattern, width, height=180):
    img = np.zeros((height, width, 3), dtype=np.uint8)

    center_x = width // 2

    color_bgr = {}
    for color_name in pattern.colors:
        ranges = pattern.color_ranges.get(color_name, [])
        if ranges:
            lower, upper = ranges[0] 
            h = (lower[0] + upper[0]) / 2
            s = (lower[1] + upper[1]) / 2
            v = (lower[2] + upper[2]) / 2
            color_bgr[color_name] = hsv_to_bgr(h, s, v)
        else:
            color_bgr[color_name] = (128, 128, 128)  

    y1 = int(height * 0.2)
    y2 = int(height * 0.8)

    MM_TO_PX_DISPLAY = width / 300.0

    for i, color in enumerate(pattern.colors):
        offset_mm = pattern.expected_positions_mm[color]
        offset_px = int(offset_mm * MM_TO_PX_DISPLAY)
        
        x_center = center_x - offset_px
        
        line_width = pattern.expected_widths[i]

        cv2.rectangle(
            img,
            (x_center - line_width // 2, y1),
            (x_center + line_width // 2, y2),
            color_bgr[color],
            -1
        )

    # axa centrală (doar referință vizuală)
    cv2.line(
        img,
        (center_x, 0),
        (center_x, height),
        (120, 120, 120),
        2
    )

    return img


class TireQCViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Tire Quality Control")
        self.root.geometry("1200x700")
        self.root.configure(bg="#2b2b2b")

        self.checker = AdvancedTireQualityChecker()
        self.checker.set_current_pattern("YAWG")
        pattern = self.checker.current_pattern

        color_bgr = {}
        for color_name in pattern.colors:
            ranges = pattern.color_ranges.get(color_name, [])
            if ranges:
                lower, upper = ranges[0]
                h = (lower[0] + upper[0]) / 2
                s = (lower[1] + upper[1]) / 2
                v = (lower[2] + upper[2]) / 2
                color_bgr[color_name] = hsv_to_bgr(h, s, v)
            else:
                color_bgr[color_name] = (128, 128, 128)

        color_map = {}
        for color in pattern.colors:
            b, g, r = color_bgr[color]
            color_map[color] = f"#{r:02x}{g:02x}{b:02x}"

        main = tk.Frame(root, bg="#2b2b2b")
        main.grid(row=0, column=0, sticky="nsew")

        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        pattern_frame = tk.Frame(main, bg="#2b2b2b")
        pattern_frame.grid(row=0, column=0, sticky="ew", pady=(10, 5))

        pattern_img = generate_pattern_image(pattern, width=900, height=180)
        pattern_img = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2RGB)
        self.pattern_tk = ImageTk.PhotoImage(Image.fromarray(pattern_img))

        self.pattern_label = tk.Label(
            pattern_frame,
            image=self.pattern_tk,
            bg="#2b2b2b",
            bd=2,
            relief="solid"
        )
        self.pattern_label.grid(row=0, column=0)

        content = tk.Frame(main, bg="#2b2b2b")
        content.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=0)

        self.video_label = tk.Label(
            content,
            bg="#1a1a1a",
            bd=2,
            relief="solid"
        )
        self.video_label.grid(row=0, column=0, sticky="n")

        info = tk.Frame(content, bg="#2b2b2b")
        info.grid(row=0, column=1, sticky="n", padx=(20, 0))

        tk.Label(
            info,
            text=f"Pattern: {pattern.name}",
            font=("Segoe UI", 14, "bold"),
            fg="white",
            bg="#2b2b2b"
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        tk.Label(
            info,
            text="Culori:",
            font=("Segoe UI", 11, "bold"),
            fg="white",
            bg="#2b2b2b"
        ).grid(row=1, column=0, sticky="w")

        for i, color in enumerate(pattern.colors):
            row = tk.Frame(info, bg="#2b2b2b")
            row.grid(row=2 + i, column=0, sticky="w", pady=4)

            c = tk.Canvas(row, width=20, height=20, bg="#2b2b2b", highlightthickness=0)
            c.grid(row=0, column=0, padx=(0, 8))
            c.create_rectangle(2, 2, 18, 18, fill=color_map[color])

            tk.Label(
                row,
                text=color.upper(),
                font=("Segoe UI", 10),
                fg="white",
                bg="#2b2b2b"
            ).grid(row=0, column=1, sticky="w")

        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps < 1:
            self.fps = 25

        self.delay = int(1000 / self.fps)
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (900, 500))

        img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.video_label.configure(image=img)
        self.video_label.image = img

        self.root.after(self.delay, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = TireQCViewer(root)
    root.mainloop()
