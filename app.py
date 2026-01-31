import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from advanced_tire_qc import AdvancedTireQualityChecker
import colorsys

# ================= CONFIG =================
# SOURCE: "local" sau "rtsp"
SOURCE = "local"   # "local" | "rtsp"

# Video local
VIDEO_PATH = r"C:\\Users\\Antonia\\Downloads\\V20251202_105058_001.avi"

# RTSP stream
RTSP_URL = "rtsp://user:pass@ip:port/stream"
FRAME_WAIT = 30  # warmup frames pentru RTSP
# ==========================================

def hsv_to_bgr(h, s, v):
    h_norm = h / 180.0
    s_norm = s / 255.0
    v_norm = v / 255.0
    
    r, g, b = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)
    return (int(b * 255), int(g * 255), int(r * 255))

def generate_pattern_image(pattern, width, height, roi, frame_size, center_x_abs):
    """
    Generează pattern preview care se ALINIAZĂ cu video-ul rescalat.
    
    Args:
        pattern: Pattern object cu colors, expected_positions_mm etc.
        width: lățimea imaginii pattern (= VIDEO_WIDTH)
        height: înălțimea imaginii pattern
        roi: (y1, y2, x1, x2) - ROI din frame original
        frame_size: (frame_width, frame_height) - dimensiunea frame-ului original
        center_x_abs: poziția X absolută a centrului în frame original
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    MM_TO_PX = 3.2
    
    # Calculăm factorul de scalare (frame original -> video rescalat)
    frame_w, frame_h = frame_size
    scale_x = width / frame_w
    
    # Poziția centrului în imaginea rescalată
    center_x_scaled = int(center_x_abs * scale_x)
    
    # ROI rescalat (pentru a desena zona activă)
    y1_roi, y2_roi, x1_roi, x2_roi = roi
    x1_scaled = int(x1_roi * scale_x)
    x2_scaled = int(x2_roi * scale_x)

    # Culori BGR din HSV ranges
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

    # Zona de desenare pe Y
    y1_draw = 0
    y2_draw = height

    # Fundal ușor mai închis pentru zona ROI
    cv2.rectangle(img, (x1_scaled, 0), (x2_scaled, height), (30, 30, 30), -1)

    # Desenăm linia de CENTRU (magenta, punctată)
    for yy in range(0, height, 6):
        cv2.line(img, (center_x_scaled, yy), (center_x_scaled, min(yy + 3, height)), (255, 0, 255), 2)
    cv2.putText(img, "CENTRU", (center_x_scaled - 30, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    # Desenăm fiecare culoare la poziția sa exactă (scalată la fel ca video-ul)
    for i, color in enumerate(pattern.colors):
        dist_mm = pattern.expected_positions_mm.get(color, 0)
        dist_px_original = int(dist_mm * MM_TO_PX)  # distanța în pixeli în frame original
        
        # Poziția în frame original (la stânga centrului)
        pos_x_original = center_x_abs - dist_px_original
        
        # Poziția scalată (aceeași scalare ca video-ul)
        pos_x_scaled = int(pos_x_original * scale_x)
        
        # Lățimea benzii (din pattern, scalată)
        if i < len(pattern.expected_widths):
            band_width_original = pattern.expected_widths[i]
        else:
            band_width_original = 6
        band_width_scaled = max(4, int(band_width_original * scale_x))
        
        half_w = band_width_scaled // 2
        bx1 = pos_x_scaled - half_w
        bx2 = pos_x_scaled + half_w

        # Desenăm banda colorată (pe toată înălțimea, fără etichete)
        cv2.rectangle(img, (bx1, y1_draw), (bx2, y2_draw), color_bgr[color], -1)

    # Indicator ROI
    cv2.putText(img, "ROI", (x1_scaled + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.line(img, (x1_scaled, 0), (x1_scaled, height), (0, 255, 255), 1)
    cv2.line(img, (x2_scaled, 0), (x2_scaled, height), (0, 255, 255), 1)

    return img


class TireQCViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Tire Quality Control")
        self.root.geometry("1400x900")
        self.root.configure(bg="#2b2b2b")

        self.checker = AdvancedTireQualityChecker()
        self.checker.set_current_pattern("YAWG")  # Schimbat la BGWY ca în run_video_analysis
        self.checker.fixed_tire_center_x = 991  # Setat ca în run_video_analysis
        self.checker.debug_mode = True
        pattern = self.checker.current_pattern

        # Dimensiuni fixe pentru video (nu se micșorează)
        self.VIDEO_WIDTH = 1000
        self.VIDEO_HEIGHT = 600
        self.roi = (299, 779, 666, 1313)  # ROI ca în run_video_analysis
        
        # Dimensiunea frame-ului original (vom actualiza după prima citire)
        self.frame_size = (1920, 1080)  # default, se va actualiza

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
        main.grid_columnconfigure(0, weight=0)  # Video fix
        main.grid_columnconfigure(1, weight=1)  # Info se extinde

        pattern_frame = tk.Frame(main, bg="#2b2b2b")
        pattern_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 5))

        # Pattern image - va fi actualizat după prima citire a frame-ului
        # pentru a folosi dimensiunile reale ale video-ului
        self.pattern_frame_widget = pattern_frame
        self.pattern_label = tk.Label(
            pattern_frame,
            bg="#2b2b2b",
            bd=2,
            relief="solid"
        )
        self.pattern_label.grid(row=0, column=0)
        self.pattern_image_created = False
        self.pattern_center_x = None  # Centrul folosit pentru pattern-ul curent

        content = tk.Frame(main, bg="#2b2b2b")
        content.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

        # Video are dimensiune FIXĂ, info panel se adaptează
        content.grid_columnconfigure(0, weight=0, minsize=self.VIDEO_WIDTH)
        content.grid_columnconfigure(1, weight=1)

        self.video_label = tk.Label(
            content,
            bg="#1a1a1a",
            bd=2,
            relief="solid",
            width=self.VIDEO_WIDTH,
            height=self.VIDEO_HEIGHT
        )
        self.video_label.grid(row=0, column=0, sticky="nw")

        info = tk.Frame(content, bg="#2b2b2b")
        info.grid(row=0, column=1, sticky="n", padx=(20, 0))

        # Pattern selector
        tk.Label(
            info,
            text="Pattern:",
            font=("Segoe UI", 11, "bold"),
            fg="white",
            bg="#2b2b2b"
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.pattern_var = tk.StringVar(value=pattern.name)
        self.pattern_selector = ttk.Combobox(
            info,
            textvariable=self.pattern_var,
            values=list(self.checker.patterns.keys()),
            state="readonly",
            width=15,
            font=("Segoe UI", 10)
        )
        self.pattern_selector.grid(row=1, column=0, sticky="w", pady=(0, 10))
        self.pattern_selector.bind("<<ComboboxSelected>>", self.on_pattern_change)

        tk.Label(
            info,
            text="Culori:",
            font=("Segoe UI", 11, "bold"),
            fg="white",
            bg="#2b2b2b"
        ).grid(row=2, column=0, sticky="w")

        # Frame pentru culori (va fi actualizat dinamic)
        self.colors_frame = tk.Frame(info, bg="#2b2b2b")
        self.colors_frame.grid(row=3, column=0, sticky="w")
        self._update_colors_display(pattern, color_map)

        self.status_label = tk.Label(
            info,
            text="Status: Necunoscut",
            font=("Segoe UI", 12, "bold"),
            fg="yellow",
            bg="#2b2b2b"
        )
        self.status_label.grid(row=4, column=0, sticky="w", pady=(10, 5))

        self.quality_label = tk.Label(
            info,
            text="Calitate: Necunoscută",
            font=("Segoe UI", 10),
            fg="white",
            bg="#2b2b2b"
        )
        self.quality_label.grid(row=5, column=0, sticky="w", pady=(0, 5))

        self.defects_label = tk.Label(
            info,
            text="Defecte: Niciunul",
            font=("Segoe UI", 10),
            fg="white",
            bg="#2b2b2b"
        )
        self.defects_label.grid(row=6, column=0, sticky="w", pady=(0, 5))

        # Deschidem captura în funcție de SOURCE
        if SOURCE == "local":
            self.cap = cv2.VideoCapture(VIDEO_PATH)
            if not self.cap.isOpened():
                raise RuntimeError(f"❌ Nu pot deschide video local: {VIDEO_PATH}")
            print(f"✅ Video local deschis: {VIDEO_PATH}")
        elif SOURCE == "rtsp":
            self.cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                raise RuntimeError(f"❌ Nu pot deschide RTSP: {RTSP_URL}")
            print(f"✅ Stream RTSP deschis: {RTSP_URL}")
            # Warmup frames pentru stabilizare RTSP
            for _ in range(FRAME_WAIT):
                self.cap.read()
        else:
            raise ValueError("SOURCE trebuie sa fie 'local' sau 'rtsp'")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps < 1:
            self.fps = 25

        self.delay = int(1000 / self.fps)
        self.update_frame()

    def _get_color_map(self, pattern):
        """Generează color_map pentru un pattern."""
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
        return color_map

    def _update_colors_display(self, pattern, color_map):
        """Actualizează afișarea culorilor în panel."""
        # Șterge widget-urile vechi
        for widget in self.colors_frame.winfo_children():
            widget.destroy()

        # Creează noile widget-uri pentru culori
        for i, color in enumerate(pattern.colors):
            row = tk.Frame(self.colors_frame, bg="#2b2b2b")
            row.grid(row=i, column=0, sticky="w", pady=4)

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

    def on_pattern_change(self, event=None):
        """Callback când se schimbă pattern-ul selectat."""
        new_pattern_name = self.pattern_var.get()
        print(f"🔄 Schimbare pattern: {new_pattern_name}")

        # Setează noul pattern
        self.checker.set_current_pattern(new_pattern_name)
        pattern = self.checker.current_pattern

        # Reset position history pentru noul pattern
        self.checker.last_positions = {}
        self.checker.shift_persistence = {}
        for color in pattern.colors:
            if color not in self.checker.position_history:
                from collections import deque
                self.checker.position_history[color] = deque(maxlen=12)
            else:
                self.checker.position_history[color].clear()

        # Actualizează afișarea culorilor
        color_map = self._get_color_map(pattern)
        self._update_colors_display(pattern, color_map)

        # Regenerează pattern image
        self.pattern_center_x = self.checker.fixed_tire_center_x
        pattern_img = generate_pattern_image(
            pattern,
            width=self.VIDEO_WIDTH,
            height=300,
            roi=self.roi,
            frame_size=self.frame_size,
            center_x_abs=self.pattern_center_x
        )
        pattern_img = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2RGB)
        self.pattern_tk = ImageTk.PhotoImage(Image.fromarray(pattern_img))
        self.pattern_label.configure(image=self.pattern_tk)
        self.pattern_label.image = self.pattern_tk

        print(f"✅ Pattern schimbat la: {new_pattern_name} ({len(pattern.colors)} culori)")

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("⚠ Frame lipsă RTSP")
                self.root.after(50, self.update_frame)
                return
            
            # La prima rulare, creăm pattern image cu dimensiunile reale ale frame-ului
            if not self.pattern_image_created:
                self.frame_size = (frame.shape[1], frame.shape[0])
                self.pattern_center_x = self.checker.fixed_tire_center_x
                pattern_img = generate_pattern_image(
                    self.checker.current_pattern,
                    width=self.VIDEO_WIDTH,
                    height=300,
                    roi=self.roi,
                    frame_size=self.frame_size,
                    center_x_abs=self.pattern_center_x
                )
                pattern_img = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2RGB)
                self.pattern_tk = ImageTk.PhotoImage(Image.fromarray(pattern_img))
                self.pattern_label.configure(image=self.pattern_tk)
                self.pattern_label.image = self.pattern_tk
                self.pattern_image_created = True
                
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
                # Prefer dynamic detected center if available
                dyn_c = debug_info.get("_detected_center")
                if dyn_c is not None:
                    center_abs = x1 + int(dyn_c)
                else:
                    center_abs = self.checker.fixed_tire_center_x

                measured_offset_mm = abs(abs_x - center_abs) / MM_TO_PX
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

            # ========== DESENĂM POZIȚIILE AȘTEPTATE (GHOST BANDS) ==========
            # Determinăm centrul folosit pentru verificare (poziție absolută în frame)
            dyn_c = debug_info.get("_detected_center")
            if dyn_c is not None:
                center_used = x1 + int(dyn_c)
            else:
                center_used = self.checker.fixed_tire_center_x

            # Actualizăm pattern-ul dacă centrul s-a mutat semnificativ (>10px)
            if self.pattern_center_x is not None and abs(center_used - self.pattern_center_x) > 10:
                self.pattern_center_x = center_used
                pattern_img = generate_pattern_image(
                    self.checker.current_pattern,
                    width=self.VIDEO_WIDTH,
                    height=300,
                    roi=self.roi,
                    frame_size=self.frame_size,
                    center_x_abs=self.pattern_center_x
                )
                pattern_img = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2RGB)
                self.pattern_tk = ImageTk.PhotoImage(Image.fromarray(pattern_img))
                self.pattern_label.configure(image=self.pattern_tk)
                self.pattern_label.image = self.pattern_tk
                print(f"🔄 Pattern actualizat - centru nou: {self.pattern_center_x}px")

            # Culorile pentru ghost bands (semi-transparente)
            ghost_colors = {
                "green": (0, 180, 0),
                "white": (200, 200, 200),
                "yellow": (0, 200, 200),
                "aqua": (200, 200, 0)
            }

            # DEBUG: afișăm pozițiile calculate (doar o dată la 30 frame-uri)
            if not hasattr(self, '_frame_counter'):
                self._frame_counter = 0
            self._frame_counter += 1
            
            # Desenăm fiecare bandă așteptată ca un dreptunghi semitransparent
            ghost_overlay = overlay.copy()
            for i, color in enumerate(self.checker.current_pattern.colors):
                expected_mm = self.checker.current_pattern.expected_positions_mm.get(color, 0)
                expected_px = int(expected_mm * MM_TO_PX)
                
                # Poziția așteptată e la STÂNGA centrului
                expected_x = center_used - expected_px
                
                # Lățimea așteptată (din pattern)
                if i < len(self.checker.current_pattern.expected_widths):
                    band_width = self.checker.current_pattern.expected_widths[i]
                else:
                    band_width = 6
                
                half_w = band_width // 2
                bx1 = expected_x - half_w
                bx2 = expected_x + half_w
                
                # Desenăm dreptunghiul ghost (doar în zona ROI pe Y)
                cv2.rectangle(ghost_overlay, (bx1, y1), (bx2, y2), ghost_colors.get(color, (128, 128, 128)), -1)
                
                # Linie centrală a benzii așteptate (punctată)
                for yy in range(y1, y2, 8):
                    cv2.line(ghost_overlay, (expected_x, yy), (expected_x, min(yy + 4, y2)), ghost_colors.get(color, (128, 128, 128)), 1)
                
                # Etichetă mică cu distanța
                cv2.putText(ghost_overlay, f"{expected_mm}mm", (bx1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, ghost_colors.get(color, (128, 128, 128)), 1)

            # Combinăm ghost overlay cu overlay principal (semi-transparent)
            overlay = cv2.addWeighted(overlay, 0.7, ghost_overlay, 0.3, 0)

            # ========== DESENĂM LINIILE DETECTATE ==========
            for color, info in result.detected_lines.items():
                x, y, w, h = info["bounding_box"]
                cx = info["x_position"]
                cy = info["y_position"]
                cv2.rectangle(overlay, (x1 + x, y1 + y), (x1 + x + w, y1 + y + h), (255, 0, 0), 2)
                cv2.circle(overlay, (x1 + cx, y1 + cy), 5, (0, 0, 255), -1)

            # Draw fixed center (fallback)
            if self.checker.fixed_tire_center_x is not None:
                cv2.line(overlay, (self.checker.fixed_tire_center_x, 0), (self.checker.fixed_tire_center_x, frame.shape[0]), (0, 255, 0), 2)

            # Draw dynamic detected center (if available)
            dyn_center = debug_info.get("_detected_center")
            if dyn_center is not None:
                cx_abs = x1 + int(dyn_center)
                cv2.line(overlay, (cx_abs, 0), (cx_abs, frame.shape[0]), (255, 0, 255), 2)
                cv2.putText(overlay, "DYN_C", (cx_abs + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

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
            overlay = cv2.resize(overlay, (self.VIDEO_WIDTH, self.VIDEO_HEIGHT))
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
