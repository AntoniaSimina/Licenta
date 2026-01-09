import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from advanced_tire_qc import AdvancedTireQualityChecker
import colorsys

VIDEO_PATH = "video_linie_galbena_margini_neregulate.avi"

CAMERA_CONFIG = {
    "IP": "10.10.10.10",
    "PORT": "554",
    "PROTOCOL": "rtsp",
    "STREAM_PATH": "/stream1",
    "USERNAME": "admin",
    "PASSWORD": "parola"
}
def hsv_to_bgr(h, s, v):
    h_norm = h / 180.0
    s_norm = s / 255.0
    v_norm = v / 255.0
    
    r, g, b = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)
    return (int(b * 255), int(g * 255), int(r * 255))

def generate_pattern_image(pattern, width, height=180):
    """Genereaza o reprezentare simpla orientativa a pattern-ului cu linii separate uniform"""
    img = np.zeros((height, width, 3), dtype=np.uint8)

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

    # Distribuire uniforma pe orizontala (orientativ, nu pozitii exacte)
    num_lines = len(pattern.colors)
    line_width = 40  # Latime fixa pentru vizibilitate
    spacing = (width - (num_lines * line_width)) // (num_lines + 1)

    for i, color in enumerate(pattern.colors):
        x_start = spacing + i * (line_width + spacing)
        x_end = x_start + line_width

        cv2.rectangle(
            img,
            (x_start, y1),
            (x_end, y2),
            color_bgr[color],
            -1
        )
        
        # Eticheta cu numele culorii
        cv2.putText(
            img,
            color.upper(),
            (x_start, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_bgr[color],
            1
        )

    return img


class TireQCViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Tire Quality Control")
        self.root.geometry("1200x700")
        self.root.configure(bg="#2b2b2b")

        self.checker = AdvancedTireQualityChecker()
        self.checker.set_current_pattern("YAWG")  # Schimbat la BGWY ca în run_video_analysis
        self.checker.fixed_tire_center_x = 991  # Setat ca în run_video_analysis
        self.checker.debug_mode = True
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

        self.status_label = tk.Label(
            info,
            text="Status: Necunoscut",
            font=("Segoe UI", 12, "bold"),
            fg="yellow",
            bg="#2b2b2b"
        )
        self.status_label.grid(row=6, column=0, sticky="w", pady=(10, 5))

        self.quality_label = tk.Label(
            info,
            text="Calitate: Necunoscută",
            font=("Segoe UI", 10),
            fg="white",
            bg="#2b2b2b"
        )
        self.quality_label.grid(row=7, column=0, sticky="w", pady=(0, 5))

        self.defects_label = tk.Label(
            info,
            text="Defecte: Niciunul",
            font=("Segoe UI", 10),
            fg="white",
            bg="#2b2b2b"
        )
        self.defects_label.grid(row=8, column=0, sticky="w", pady=(0, 5))

        # Construiește URL-ul RTSP din configurație
        rtsp_url = f"{CAMERA_CONFIG['PROTOCOL']}://{CAMERA_CONFIG['USERNAME']}:{CAMERA_CONFIG['PASSWORD']}@{CAMERA_CONFIG['IP']}:{CAMERA_CONFIG['PORT']}{CAMERA_CONFIG['STREAM_PATH']}"
        
        # Încearcă să se conecteze la camera RTSP
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        # Fallback la fișierul video local dacă RTSP nu funcționează
        if not self.cap.isOpened():
            print(f"⚠ Nu pot deschide stream-ul RTSP: {rtsp_url}")
            print(f"Folosesc fallback la fișierul video local: {VIDEO_PATH}")
            self.cap = cv2.VideoCapture(VIDEO_PATH)
            
        if not self.cap.isOpened():
            raise RuntimeError("Nu pot deschide nici stream-ul RTSP, nici fișierul video local")


        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps < 1:
            self.fps = 25

        self.delay = int(1000 / self.fps)
        self.roi = (299, 779, 666, 1313)  # ROI ca în run_video_analysis
        self.update_frame()

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("⚠ Frame lipsă RTSP")
                self.root.after(50, self.update_frame)
                return
                
            if self.roi:
                y1, y2, x1, x2 = self.roi
                frame_roi = frame[y1:y2, x1:x2]
            else:
                frame_roi = frame
                x1, y1 = 0, 0

            result = self.checker.analyze_tire_frame(frame_roi)

            defects_abs, debug_info = self.checker._analyze_frame_absolute(frame_roi, tire_center_x=self.checker.fixed_tire_center_x - x1, x_offset=x1)
            
            # VERIFICARE IMEDIATA a pozitiilor (ca in analyze_video)
            from advanced_tire_qc import DefectType, DefectReport
            MM_TO_PX = 3.2
            for color, info in result.detected_lines.items():
                abs_x = info["x_position"] + x1
                measured_offset_mm = abs(abs_x - self.checker.fixed_tire_center_x) / MM_TO_PX
                expected_offset_mm = self.checker.current_pattern.expected_positions_mm[color]
                delta_mm = abs(measured_offset_mm - expected_offset_mm)

                if delta_mm > 10.0:
                    result.defects.append(
                        DefectReport(
                            defect_type=DefectType.LINE_SHIFTED,
                            severity=min(delta_mm / 20.0, 1.0),
                            position=(info["x_position"], info["y_position"]),
                            description=f"{color} POZITIE GRESITA: {measured_offset_mm:.1f}mm (asteptat {expected_offset_mm:.1f}mm, delta {delta_mm:.1f}mm)",
                            confidence=0.95
                        )
                    )
            
            for d in defects_abs:
                result.defects.append(d)

            status_message, quality_level, is_valid, summary = self.checker._generate_status_messages(
                {c: c in result.detected_lines for c in self.checker.current_pattern.colors},
                result.defects
            )
            result.status_message = status_message
            result.quality_level = quality_level
            result.is_valid = is_valid
            result.summary = summary

            overlay = frame.copy()
            if self.roi:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

            for color, info in result.detected_lines.items():
                x, y, w, h = info["bounding_box"]
                cx = info["x_position"]
                cy = info["y_position"]
                cv2.rectangle(overlay, (x1 + x, y1 + y), (x1 + x + w, y1 + y + h), (255, 0, 0), 2)
                cv2.circle(overlay, (x1 + cx, y1 + cy), 5, (0, 0, 255), -1)

            cv2.line(overlay, (self.checker.fixed_tire_center_x, 0), (self.checker.fixed_tire_center_x, frame.shape[0]), (0, 255, 0), 2)

            for defect in result.defects:
                dx = x1 + defect.position[0]
                dy = y1 + defect.position[1]
                col = (0, 0, 255) if defect.severity > 0.7 else ((0, 165, 255) if defect.severity > 0.3 else (0, 255, 255))
                cv2.circle(overlay, (dx, dy), 10, col, 2)

            verdict_color = (0, 255, 0) if result.is_valid else (0, 0, 255)
            cv2.putText(overlay, result.quality_level, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, verdict_color, 2)
            cv2.putText(overlay, result.status_message, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, verdict_color, 1)

            self.status_label.config(text=f"Status: {result.status_message}", fg="green" if result.is_valid else "red")
            self.quality_label.config(text=f"Calitate: {result.quality_level}")
            defects_text = f"Defecte: {len(result.defects)}" if result.defects else "Defecte: Niciunul"
            self.defects_label.config(text=defects_text)

            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            overlay = cv2.resize(overlay, (900, 500))
            img = ImageTk.PhotoImage(Image.fromarray(overlay))
            self.video_label.configure(image=img)
            self.video_label.image = img

            self.root.after(self.delay, self.update_frame)
        except Exception as e:
            print("EROARE LIVE:", e)
            self.root.after(100, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = TireQCViewer(root)
    root.mainloop()
