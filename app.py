import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import os
import json
from advanced_tire_qc import AdvancedTireQualityChecker, SCALE_FINAL, OFFSET_FINAL, WARPED_SIZE
import colorsys

# ================= CONFIG =================
# ROI partajat cu alte scripturi (y1, y2, x1, x2)
# ROI = (233, 659, 379, 807)  # ROI ca în run_video_analysis CELELALTE 2, pt video 1 => (379, 875, 680, 1294)
ROI = (289, 479, 390, 723) # pt video 1 => (379, 875, 680, 1294)
# ROI este definit pe frame-ul original (înainte de warp).
ROI_SPACE = "raw"  # "raw" | "warped"
# SOURCE: "local" sau "rtsp"
SOURCE = "local"   # "local" | "rtsp"

# Video local

# VIDEO_PATH = r"C:\Users\Lenovo\Downloads\files\V20260129_153506_001.avi" #GYRL
VIDEO_PATH = r"C:\Users\Lenovo\Downloads\files\V20260219_123605_001.avi" #WWAA
# VIDEO_PATH = r"C:\Users\Lenovo\Downloads\files\V20260212_085654_001.avi" #WYO
# VIDEO_PATH = r"C:\Users\Lenovo\Downloads\files\V20260219_133420_001.avi" #WAR
# VIDEO_PATH = r"C:\Users\Lenovo\Downloads\files\V20260219_133420_001.avi" #WAL
# VIDEO_PATH = r"C:\Users\Lenovo\Downloads\files\V20260219_081539_001.avi" #ARRY
# VIDEO_PATH = r"C:\Users\Lenovo\Downloads\files\V20260219_151249_001.avi" #LAW
# VIDEO_PATH = r"C:\Users\Lenovo\Downloads\files\V20260129_153301_001.avi" #FINAL CHALLANGE

# RTSP stream
RTSP_URL = "rtsp://user:pass@ip:port/stream"
FRAME_WAIT = 30  # warmup frames pentru RTSP

# Fișierul JSON cu pattern-urile de producție
PATTERNS_JSON_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patterns_productie.json")

# Fișiere de calibrare
CAMERA_CALIBRATION_FILE = "calibrare_camera.npz"
HOMOGRAPHY_FILE = "matrice_omografie.npy"
# ==========================================

def hsv_to_bgr(h, s, v):
    h_norm = h / 180.0
    s_norm = s / 255.0
    v_norm = v / 255.0
    
    r, g, b = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)
    return (int(b * 255), int(g * 255), int(r * 255))


def load_calibration():
    """Încarcă fișierele de calibrare (camera matrix și homography matrix)."""
    camera_mtx = None
    dist = None

    if os.path.exists(CAMERA_CALIBRATION_FILE):
        data = np.load(CAMERA_CALIBRATION_FILE)
        camera_mtx = data["mtx"]
        dist = data["dist"]
        print(f"[INFO] Calibrare camera încărcată din {CAMERA_CALIBRATION_FILE}")
    else:
        print(f"[WARN] Fișier lipsă: {CAMERA_CALIBRATION_FILE}. Voi continua fără undistort.")

    if not os.path.exists(HOMOGRAPHY_FILE):
        print(f"[WARN] Fișier lipsă: {HOMOGRAPHY_FILE}. Voi folosi imaginea originală.")
        homography = None
    else:
        homography = np.load(HOMOGRAPHY_FILE)
        print(f"[INFO] Omografie încărcată din {HOMOGRAPHY_FILE}")

    return camera_mtx, dist, homography


def preprocess_frame(frame, camera_mtx, dist, homography):
    """Preprocessează frame-ul: undistort + perspectivă warping."""
    processed = frame

    # Aplică undistortion dacă dispunem de calibrare camerei
    if camera_mtx is not None and dist is not None:
        h, w = frame.shape[:2]
        new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(camera_mtx, dist, (w, h), 1, (w, h))
        processed = cv2.undistort(frame, camera_mtx, dist, None, new_camera_mtx)

    # Aplică perspectivă warping dacă dispunem de matrice omografie
    if homography is not None:
        processed = cv2.warpPerspective(processed, homography, WARPED_SIZE)
    
    return processed


def project_roi_raw_to_warped(roi, homography, warped_size):
    """Proiectează ROI din coordonate frame original în coordonate warped."""
    if homography is None or roi is None or len(roi) != 4:
        return roi

    y1, y2, x1, x2 = roi
    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32
    ).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    xs = warped_corners[:, 0]
    ys = warped_corners[:, 1]

    wx, wy = warped_size
    rx1 = int(max(0, min(np.floor(xs.min()), wx - 1)))
    rx2 = int(max(0, min(np.ceil(xs.max()), wx)))
    ry1 = int(max(0, min(np.floor(ys.min()), wy - 1)))
    ry2 = int(max(0, min(np.ceil(ys.max()), wy)))

    return (ry1, ry2, rx1, rx2)


def load_pattern_names_from_json(json_path):
    """Returneaza lista unica de pattern_name, in ordinea din JSON."""
    names = []
    seen = set()

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[WARN] Nu pot citi pattern-uri din JSON: {exc}")
        return names

    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("pattern_name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)

    return names


def get_expected_mm_by_color_index(pattern, color_name, index_in_colors):
    """Returnează poziția în mm pentru o culoare la indexul dat din pattern.colors."""
    mm_list = getattr(pattern, "expected_positions_mm_by_index", None)
    if isinstance(mm_list, list) and index_in_colors < len(mm_list):
        return int(mm_list[index_in_colors])

    return pattern.expected_positions_mm.get(color_name, 0)


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
    # Pentru pattern-uri cu culori duplicate, folosim indexul din pattern.colors.
    for i, color in enumerate(pattern.colors):
        # Folosim funcția pentru a obține poziția corectă pentru această culoare la acest index
        dist_mm = get_expected_mm_by_color_index(pattern, color, i)
        dist_px_warped = int(dist_mm * SCALE_FINAL + OFFSET_FINAL)
        
        # Poziția în frame-ul warped (la stânga centrului, identic cu overlay-ul video)
        pos_x_warped = center_x_abs - dist_px_warped
        
        # Poziția scalată (aceeași scalare ca video-ul rescalat)
        pos_x_scaled = int(pos_x_warped * scale_x)
        
        # Lățimea benzii (din pattern, scalată)
        if i < len(pattern.expected_widths):
            band_width_warped = pattern.expected_widths[i]
        else:
            band_width_warped = 6
        band_width_scaled = max(6, int(band_width_warped * scale_x))
        
        half_w = band_width_scaled // 2
        bx1 = pos_x_scaled - half_w
        bx2 = pos_x_scaled + half_w

        cv2.rectangle(img, (bx1, y1_draw), (bx2, y2_draw), color_bgr[color], -1)
        idx_text = str(i + 1)
        tx = pos_x_scaled - 4
        ty = height // 2 + 5
        cv2.putText(img, idx_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, idx_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        mm_text = f"{dist_mm}mm"
        cv2.putText(img, mm_text, (pos_x_scaled - 16, height - 6 - (i % 2) * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, mm_text, (pos_x_scaled - 16, height - 6 - (i % 2) * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1, cv2.LINE_AA)

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

        # Încarcă calibrarea (camera matrix și homography)
        self.camera_mtx, self.dist, self.homography = load_calibration()

        # Încarcă pattern-urile din fișierul JSON de producție
        self.checker = AdvancedTireQualityChecker(patterns_json_file=PATTERNS_JSON_FILE)
        
        # Selectează primul pattern disponibil (sau un pattern specific dacă există)
        available_patterns = self.checker.get_pattern_names()
        if available_patterns:
            # Setează primul pattern ca implicit
            default_pattern = available_patterns[0]
            self.checker.set_current_pattern(default_pattern)
            print(f"✅ Pattern implicit setat: {default_pattern}")
        else:
            print("⚠️ Niciun pattern disponibil!")
        
        self.checker.fixed_tire_center_x = 991  # Setat ca în run_video_analysis
        self.checker.debug_mode = True
        pattern = self.checker.current_pattern
        
        # Verificare de siguranță în cazul în care nu există pattern-uri
        if pattern is None:
            raise RuntimeError("Nu s-a putut încărca niciun pattern. Verificați fișierul patterns_productie.json")

        # Dimensiuni fixe pentru video (nu se micșorează)
        self.VIDEO_WIDTH = 1000
        self.VIDEO_HEIGHT = 600
        # Folosim ROI-ul partajat de nivel modul
        self.roi = ROI
        self.roi_space = ROI_SPACE
        print(f"[INFO] ROI activ ({self.roi_space}): {self.roi}")

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
        content.grid_rowconfigure(0, weight=1)
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

        info = tk.Frame(content, bg="#2b2b2b", width=300, height=550)
        info.grid(row=0, column=1, sticky="nw", padx=(20, 0))
        info.grid_propagate(False)  # Previne redimensionarea automată
        # info.grid_rowconfigure(3, weight=1)
        info.grid_columnconfigure(0, weight=1)

        # ============ PATTERN SELECTOR SECTION (FIXED SIZE) ============
        pattern_selector_frame = tk.Frame(info, bg="#2b2b2b", height=110, width=300)
        pattern_selector_frame.grid(row=0, column=0, sticky="nw", pady=(0, 10))
        pattern_selector_frame.grid_propagate(False)
        pattern_selector_frame.grid_columnconfigure(0, weight=1)

        tk.Label(
            pattern_selector_frame,
            text="Selectează Pattern:",
            font=("Segoe UI", 11, "bold"),
            fg="white",
            bg="#2b2b2b"
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        json_pattern_names = load_pattern_names_from_json(PATTERNS_JSON_FILE)
        valid_loaded_names = set(self.checker.get_pattern_names())
        self.all_pattern_names = [name for name in json_pattern_names if name in valid_loaded_names]
        if not self.all_pattern_names:
            self.all_pattern_names = self.checker.get_pattern_names()

        # Navigare rapida pattern-uri
        nav_frame = tk.Frame(pattern_selector_frame, bg="#2b2b2b")
        nav_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        nav_frame.grid_columnconfigure(1, weight=1)

        tk.Button(
            nav_frame,
            text="◀ Prev",
            font=("Segoe UI", 9),
            bg="#404040",
            fg="white",
            relief="flat",
            padx=5,
            command=self.select_previous_pattern
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))

        tk.Button(
            nav_frame,
            text="Next ▶",
            font=("Segoe UI", 9),
            bg="#404040",
            fg="white",
            relief="flat",
            padx=5,
            command=self.select_next_pattern
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))

        # Cautare cu autocomplete (fara popup custom)
        search_frame = tk.Frame(pattern_selector_frame, bg="#2b2b2b")
        search_frame.grid(row=2, column=0, sticky="ew")
        search_frame.grid_columnconfigure(1, weight=1)

        tk.Label(
            search_frame,
            text="Cauta:",
            font=("Segoe UI", 9),
            fg="#aaaaaa",
            bg="#2b2b2b"
        ).grid(row=0, column=0, sticky="w", padx=(0, 5))

        self.pattern_var = tk.StringVar(value=pattern.name)
        self.pattern_selector = ttk.Combobox(
            search_frame,
            textvariable=self.pattern_var,
            values=self.all_pattern_names,
            font=("Segoe UI", 10),
            state="normal"
        )
        self.pattern_selector.grid(row=0, column=1, sticky="ew")
        self.pattern_selector.bind("<KeyRelease>", self.on_pattern_search_keyrelease)
        self.pattern_selector.bind("<<ComboboxSelected>>", self.on_pattern_change)
        self.pattern_selector.bind("<Return>", self.on_pattern_search_enter)

        tk.Button(
            search_frame,
            text="Aplica",
            font=("Segoe UI", 8),
            bg="#404040",
            fg="white",
            relief="flat",
            width=7,
            command=self.on_pattern_search_enter
        ).grid(row=0, column=2, sticky="ew", padx=(5, 0))

        # ============ COLORS SECTION (FIXED SIZE) ============
        colors_container_frame = tk.Frame(info, bg="#2b2b2b", height=200, width=300)
        colors_container_frame.grid(row=1, column=0, sticky="nw", pady=(0, 10))
        colors_container_frame.grid_propagate(False)  # Nu se redimensionează automat
        colors_container_frame.grid_columnconfigure(0, weight=0)

        tk.Label(
            colors_container_frame,
            text="Culori:",
            font=("Segoe UI", 11, "bold"),
            fg="white",
            bg="#2b2b2b"
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        # Frame pentru culori (va fi actualizat dinamic)
        self.colors_frame = tk.Frame(colors_container_frame, bg="#2b2b2b")
        self.colors_frame.grid(row=1, column=0, sticky="w")
        self._update_colors_display(pattern, color_map)

        # ============ PATTERN INFO SECTION ============
        pattern_info_frame = tk.Frame(info, bg="#2b2b2b", width=300)
        pattern_info_frame.grid(row=2, column=0, sticky="nw", pady=(0, 10))
        pattern_info_frame.grid_columnconfigure(0, weight=0)

        tk.Label(
            pattern_info_frame,
            text="Informații pattern:",
            font=("Segoe UI", 11, "bold"),
            fg="white",
            bg="#2b2b2b"
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        # Pattern Name (mare, proeminent)
        self.pattern_name_label = tk.Label(
            pattern_info_frame,
            text=pattern.name,
            font=("Segoe UI", 40, "bold"),
            fg="#00ff00",  # verde deschis
            bg="#2b2b2b"
        )
        self.pattern_name_label.grid(row=1, column=0, sticky="w", pady=(0, 10))

        # Recipe ID
        self.recipe_id_label = tk.Label(
            pattern_info_frame,
            text=f"Recipe ID: {pattern.recipe_id}",
            font=("Segoe UI", 10),
            fg="#aaaaaa",
            bg="#2b2b2b"
        )
        self.recipe_id_label.grid(row=2, column=0, sticky="w", pady=(0, 2))

        # Product Code
        self.product_code_label = tk.Label(
            pattern_info_frame,
            text=f"Product Code: {pattern.product_code}",
            font=("Segoe UI", 10),
            fg="#aaaaaa",
            bg="#2b2b2b"
        )
        self.product_code_label.grid(row=3, column=0, sticky="w", pady=(0, 2))

        # Nume pattern (canonic)
        self.pattern_official_label = tk.Label(
            pattern_info_frame,
            text=f"Nume pattern: {pattern.name}",
            font=("Segoe UI", 10),
            fg="#aaaaaa",
            bg="#2b2b2b"
        )
        self.pattern_official_label.grid(row=4, column=0, sticky="w", pady=(0, 2))

        # ============ STATUS SECTION (sub content, în main) ============
        status_frame = tk.Frame(main, bg="#2b2b2b")
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        status_frame.grid_columnconfigure(0, weight=0)
        status_frame.grid_columnconfigure(1, weight=0)
        status_frame.grid_columnconfigure(2, weight=0)

        self.status_label = tk.Label(
            status_frame,
            text="Status: Necunoscut",
            font=("Segoe UI", 12, "bold"),
            fg="yellow",
            bg="#2b2b2b",
            anchor="w"
        )
        self.status_label.grid(row=0, column=0, sticky="w", padx=(0, 30))

        self.quality_label = tk.Label(
            status_frame,
            text="Calitate: Necunoscută",
            font=("Segoe UI", 10),
            fg="white",
            bg="#2b2b2b",
            wraplength=280,
            justify="left",
            anchor="w"
        )
        self.quality_label.grid(row=0, column=1, sticky="w", padx=(0, 30))

        self.defects_label = tk.Label(
            status_frame,
            text="Defecte: Niciunul",
            font=("Segoe UI", 10),
            fg="white",
            bg="#2b2b2b",
            wraplength=280,
            justify="left",
            anchor="w"
        )
        self.defects_label.grid(row=0, column=2, sticky="w")

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
        self._last_terminal_report_signature = None
        self._last_valid_state = None
        self._last_position_skip_signature = None
        self._last_offset_report_signature = None
        self.update_frame()

    def _report_result_to_terminal(self, result):
        """Afișează în terminal problemele detectate (fără spam pe fiecare frame)."""
        defect_descriptions = sorted(d.description for d in result.defects)
        signature = (result.is_valid, result.status_message, tuple(defect_descriptions))

        if signature == self._last_terminal_report_signature:
            return

        self._last_terminal_report_signature = signature

        if not result.is_valid:
            pattern_name = self.checker.current_pattern.name if self.checker.current_pattern else "N/A"
            print(f"[ALERTA] Pattern {pattern_name} | {result.quality_level} | {result.status_message}")
            if result.defects:
                for defect in result.defects:
                    print(f"  - [{defect.defect_type.value}] {defect.description}")
            else:
                print("  - Problemă detectată fără defecte detaliate.")
        elif self._last_valid_state is False:
            print("[INFO] Sistemul a revenit la stare ACCEPTAT.")

        self._last_valid_state = result.is_valid

    def _report_position_check_skips(self, debug_info):
        """Raportează explicit când verificarea de poziție a fost sărită pentru culori."""
        if not isinstance(debug_info, dict):
            return

        skipped = []
        for color, info in debug_info.items():
            if not isinstance(info, dict):
                continue
            if info.get("position_check_skipped"):
                reason = info.get("position_check_skipped_reason", "motiv_necunoscut")
                skipped.append((color, reason))

        skipped.sort(key=lambda x: x[0])
        signature = tuple(skipped)
        if signature == self._last_position_skip_signature:
            return

        self._last_position_skip_signature = signature
        if skipped:
            details = "; ".join([f"{color}: {reason}" for color, reason in skipped])
            print(f"[WARN][POS-CHECK-SKIPPED] {details}")

    def _report_line_offsets(self, result, center_px):
        """Raportează offset-ul măsurat vs așteptat pentru fiecare linie din pattern."""
        if self.checker.current_pattern is None:
            return

        rows = []
        for idx, color in enumerate(self.checker.current_pattern.colors):
            line_key = self.checker._line_key(color, idx)
            expected_mm = float(get_expected_mm_by_color_index(self.checker.current_pattern, color, idx))

            info = result.detected_lines.get(line_key)
            if info is None:
                rows.append((line_key, "MISS", None, expected_mm, None))
                continue

            abs_x = float(info["x_position"])
            measured_mm = abs(abs_x - float(center_px)) / max(float(SCALE_FINAL), 1e-6)
            delta_mm = measured_mm - expected_mm
            rows.append((line_key, "OK", measured_mm, expected_mm, delta_mm))

        signature = (
            self.checker.current_pattern.name,
            tuple(
                (
                    k,
                    st,
                    None if m is None else round(m, 1),
                    round(e, 1),
                    None if d is None else round(d, 1),
                )
                for k, st, m, e, d in rows
            ),
        )
        if signature == self._last_offset_report_signature:
            return

        self._last_offset_report_signature = signature
        print(f"[DEBUG][OFFSETS] Pattern {self.checker.current_pattern.name} | center={center_px}px")
        for line_key, state, measured_mm, expected_mm, delta_mm in rows:
            if state == "MISS":
                print(f"  - {line_key}: MISS | expected={expected_mm:.1f}mm")
            else:
                print(
                    f"  - {line_key}: measured={measured_mm:.1f}mm | "
                    f"expected={expected_mm:.1f}mm | delta={delta_mm:+.1f}mm"
                )

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

    def select_previous_pattern(self):
        """Selectează pattern-ul anterior din listă."""
        all_patterns = self.all_pattern_names
        current_name = self.checker.current_pattern.name
        
        try:
            current_idx = all_patterns.index(current_name)
            new_idx = (current_idx - 1) % len(all_patterns)
            new_pattern = all_patterns[new_idx]
            self.apply_pattern(new_pattern)
        except (ValueError, IndexError):
            pass

    def select_next_pattern(self):
        """Selectează pattern-ul următor din listă."""
        all_patterns = self.all_pattern_names
        current_name = self.checker.current_pattern.name
        
        try:
            current_idx = all_patterns.index(current_name)
            new_idx = (current_idx + 1) % len(all_patterns)
            new_pattern = all_patterns[new_idx]
            self.apply_pattern(new_pattern)
        except (ValueError, IndexError):
            pass

    def on_pattern_search_keyrelease(self, event=None):
        """Filtrează lista din combobox în timp ce scrii, fără auto-select forțat."""
        if event and event.keysym in ('Return', 'Down', 'Up', 'Left', 'Right', 'Escape', 'Tab', 'Shift_L', 'Shift_R', 'Control_L', 'Control_R'):
            return

        typed = self.pattern_var.get().strip()
        cursor_pos = self.pattern_selector.index(tk.INSERT)

        if not typed:
            self.pattern_selector["values"] = self.all_pattern_names
            return

        filtered = [p for p in self.all_pattern_names if typed.upper() in p.upper()]
        self.pattern_selector["values"] = filtered if filtered else self.all_pattern_names

        self.pattern_selector.set(typed)
        self.pattern_selector.icursor(cursor_pos)

        try:
            self.pattern_selector.tk.call('ttk::combobox::Post', self.pattern_selector)
        except Exception:
            pass

    def on_pattern_search_enter(self, event=None):
        """Aplică pattern-ul scris/selectat când apeși Enter."""
        typed = self.pattern_var.get().strip()
        if not typed:
            return

        if typed in self.checker.patterns:
            self.apply_pattern(typed)
            return

        filtered = [p for p in self.all_pattern_names if typed.upper() in p.upper()]
        if filtered:
            self.apply_pattern(filtered[0])

    def on_pattern_change(self, event=None):
        """Aplică pattern-ul selectat din dropdown."""
        selected = self.pattern_var.get().strip()
        if selected in self.checker.patterns:
            self.apply_pattern(selected)

    def apply_pattern(self, pattern_name):
        """Aplică un pattern și actualizează UI-ul."""
        if pattern_name not in self.checker.patterns:
            print(f"⚠️ Pattern '{pattern_name}' nu există")
            return
        
        print(f"✅ Pattern selectat: {pattern_name}")

        # Setează noul pattern
        self.checker.set_current_pattern(pattern_name)
        pattern = self.checker.current_pattern

        # Reset position history pentru noul pattern
        self.checker.last_positions = {}
        self.checker.shift_persistence = {}
        self._last_offset_report_signature = None
        for color in pattern.colors:
            if color not in self.checker.position_history:
                from collections import deque
                self.checker.position_history[color] = deque(maxlen=12)
            else:
                self.checker.position_history[color].clear()

        self.pattern_var.set(pattern.name)
        self.pattern_selector["values"] = self.all_pattern_names

        # Actualizează afișarea culorilor
        color_map = self._get_color_map(pattern)
        self._update_colors_display(pattern, color_map)

        # Actualizează informațiile pattern-ului
        self.pattern_name_label.config(text=pattern.name)
        self.recipe_id_label.config(text=f"Recipe ID: {pattern.recipe_id}")
        self.product_code_label.config(text=f"Product Code: {pattern.product_code}")
        self.pattern_official_label.config(text=f"Nume pattern: {pattern.name}")

        # Regenerează pattern image
        self.pattern_center_x = self.checker.fixed_tire_center_x

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

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("⚠ Frame lipsă")
                self.root.after(50, self.update_frame)
                return
            
            # Preprocess frame: undistort + warp
            frame_warped = preprocess_frame(frame, self.camera_mtx, self.dist, self.homography)
            
            if frame_warped is None or frame_warped.size == 0:
                print("⚠ Frame warping failed")
                self.root.after(50, self.update_frame)
                return
            
            # Aplică ROI în fluxul live (similar cu analyze_video).
            roi_ok = False
            y1 = y2 = x1 = x2 = 0
            frame_for_analysis = frame_warped
            if self.roi and len(self.roi) == 4:
                roi_for_warped = self.roi
                if self.roi_space == "raw":
                    roi_for_warped = project_roi_raw_to_warped(self.roi, self.homography, WARPED_SIZE)

                y1, y2, x1, x2 = roi_for_warped
                y1 = max(0, min(y1, frame_warped.shape[0]))
                y2 = max(0, min(y2, frame_warped.shape[0]))
                x1 = max(0, min(x1, frame_warped.shape[1]))
                x2 = max(0, min(x2, frame_warped.shape[1]))
                roi_ok = (y2 > y1 and x2 > x1)
                if roi_ok:
                    frame_for_analysis = frame_warped[y1:y2, x1:x2]

            # Detectează centrul dinamic în ROI-ul activ (dacă există), altfel pe cadrul complet.
            if roi_ok:
                detected_center_local = self.checker._find_center_by_intensity_profile(frame_for_analysis)
                if detected_center_local is None:
                    # Fallback corect în coordonate absolute: centrul ROI-ului proiectat.
                    detected_center = x1 + (x2 - x1) // 2
                else:
                    detected_center = x1 + detected_center_local
            else:
                detected_center = self.checker._find_center_by_intensity_profile(frame_warped)
                if detected_center is None:
                    detected_center = WARPED_SIZE[0] // 2

            # Stabilizare: 80% istoric + 20% detectat
            center_px = int(0.8 * self.checker.last_center + 0.2 * detected_center)
            self.checker.last_center = center_px
            
            roi_for_pattern = (y1, y2, x1, x2) if roi_ok else (0, WARPED_SIZE[1], 0, WARPED_SIZE[0])
            pattern_img = generate_pattern_image(
                self.checker.current_pattern,
                self.VIDEO_WIDTH, 250,
                roi_for_pattern, WARPED_SIZE, center_px
            )
            pattern_rgb = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2RGB)
            pattern_tk = ImageTk.PhotoImage(Image.fromarray(pattern_rgb))
            self.pattern_label.configure(image=pattern_tk)
            self.pattern_label.image = pattern_tk

            # Analizează frame-ul (ROI dacă este valid, altfel frame complet)
            result = self.checker.analyze_tire_frame(frame_for_analysis)

            # Dacă am analizat pe ROI, translăm coordonatele în sistemul frame-ului complet.
            if roi_ok:
                remapped_lines = {}
                for color, info in result.detected_lines.items():
                    x, y, w, h = info["bounding_box"]
                    info_abs = dict(info)
                    info_abs["bounding_box"] = (x + x1, y + y1, w, h)
                    info_abs["x_position"] = info["x_position"] + x1
                    info_abs["y_position"] = info["y_position"] + y1
                    remapped_lines[color] = info_abs
                result.detected_lines = remapped_lines

                remapped_defects = []
                for d in result.defects:
                    remapped_defects.append(
                        type(d)(
                            defect_type=d.defect_type,
                            severity=d.severity,
                            position=(d.position[0] + x1, d.position[1] + y1),
                            description=d.description,
                            confidence=d.confidence
                        )
                    )
                result.defects = remapped_defects

            # Verificare poziții față de centru folosind cotele din pattern (JSON).
            if roi_ok:
                defects_abs, debug_info_abs = self.checker._analyze_frame_absolute(
                    frame_for_analysis,
                    tire_center_x=center_px - x1,
                    x_offset=x1,
                    lock_to_input_center=True,
                )
            else:
                defects_abs, debug_info_abs = self.checker._analyze_frame_absolute(
                    frame_warped,
                    tire_center_x=center_px,
                    x_offset=0,
                    lock_to_input_center=True,
                )
            for d in defects_abs:
                result.defects.append(d)

            self._report_position_check_skips(debug_info_abs)
            
            # Rezumat status
            found_lines = {
                self.checker._line_key(c, i): (self.checker._line_key(c, i) in result.detected_lines)
                for i, c in enumerate(self.checker.current_pattern.colors)
            }
            status_message, quality_level, is_valid, summary = self.checker._generate_status_messages(
                found_lines,
                result.defects
            )
            result.status_message = status_message
            result.quality_level = quality_level
            result.is_valid = is_valid
            result.summary = summary

            self._report_line_offsets(result, center_px)

            # Raportează în terminal problemele detectate de algoritm.
            self._report_result_to_terminal(result)

            # ========== OVERLAY VIZUAL (redesigned) ==========
            overlay = frame_warped.copy()
            h_warped, w_warped = overlay.shape[:2]
            cur_pattern = self.checker.current_pattern

            # --- Layer semi-transparent: benzi colorate la pozițiile așteptate ---
            semi = overlay.copy()

            band_positions_warped = []
            for idx, color_name in enumerate(cur_pattern.colors):
                cota_mm = get_expected_mm_by_color_index(cur_pattern, color_name, idx)
                expected_px = int(cota_mm * SCALE_FINAL + OFFSET_FINAL)
                pos_x = int(center_px - expected_px)
                pos_x = max(0, min(w_warped - 1, pos_x))
                band_positions_warped.append((pos_x, color_name, cota_mm, idx))

                ranges = cur_pattern.color_ranges.get(color_name, [])
                if ranges:
                    lo, hi = ranges[0]
                    band_bgr = hsv_to_bgr((lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2)
                else:
                    band_bgr = (128, 128, 128)

                bw = max(12, cur_pattern.expected_widths[idx] if idx < len(cur_pattern.expected_widths) else 6)
                cv2.rectangle(semi, (pos_x - bw, 0), (pos_x + bw, h_warped), band_bgr, -1)

            cv2.rectangle(semi, (center_px - 6, 0), (center_px + 6, h_warped), (255, 0, 255), -1)
            overlay = cv2.addWeighted(overlay, 0.72, semi, 0.28, 0)

            # --- Elemente opace pe frame-ul warped ---
            cv2.line(overlay, (center_px, 0), (center_px, h_warped), (255, 0, 255), 3)

            for pos_x, _, _, _ in band_positions_warped:
                cv2.line(overlay, (pos_x, 0), (pos_x, h_warped), (255, 255, 255), 1)

            if roi_ok:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 4)

            for color_key, info in result.detected_lines.items():
                bx, by, bw, bh = info["bounding_box"]
                cx_det = info["x_position"]
                cy_det = info["y_position"]
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (255, 200, 0), 4)
                cv2.circle(overlay, (cx_det, cy_det), 10, (0, 0, 255), -1)
                cv2.circle(overlay, (cx_det, cy_det), 10, (255, 255, 255), 2)

            # --- Resize pentru afișaj ---
            overlay_resized = cv2.resize(overlay, (self.VIDEO_WIDTH, self.VIDEO_HEIGHT))
            sx = self.VIDEO_WIDTH / w_warped
            sy = self.VIDEO_HEIGHT / h_warped

            # --- Text clar pe frame-ul RESIZED (nu pe warped) ---
            def _put_outlined(img, text, pos, scale, fg, thickness=1):
                cv2.putText(img, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(img, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)

            # Fără banner de verdict peste video: verdictul rămâne în panoul din dreapta.
            _put_outlined(overlay_resized, f"SCALE={SCALE_FINAL:.1f}px/mm | OFS={OFFSET_FINAL}px", (15, 24), 0.38, (200, 200, 200), 1)

            # Etichete pozitii asteptate
            label_y_start = 100
            for i_bp, (pos_x_w, color_name, cota_mm, idx) in enumerate(band_positions_warped):
                lx = int(pos_x_w * sx)
                label = f"{idx+1}:{color_name} ({cota_mm}mm)"
                ly = label_y_start + i_bp * 20
                _put_outlined(overlay_resized, label, (max(5, lx + 6), ly), 0.45, (0, 255, 80), 1)

            # Label centru
            center_rx = int(center_px * sx)
            _put_outlined(overlay_resized, f"Centru: {center_px}px",
                          (max(5, center_rx - 50), self.VIDEO_HEIGHT - 15), 0.5, (255, 100, 255), 1)

            # Label ROI
            if roi_ok:
                rx1 = int(x1 * sx)
                ry1 = int(y1 * sy)
                _put_outlined(overlay_resized, "ROI", (rx1 + 4, max(18, ry1 - 6)), 0.6, (0, 255, 255), 1)

            # Contor defecte
            n_defects = len(result.defects)
            if n_defects > 0:
                _put_outlined(overlay_resized, f"Defecte: {n_defects}",
                              (self.VIDEO_WIDTH - 170, 30), 0.6, (0, 80, 255), 2)
                              
            # Redimensionează pentru afișaj
            overlay_rgb = cv2.cvtColor(overlay_resized, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(overlay_rgb))
            self.video_label.configure(image=img)
            self.video_label.image = img

            self.status_label.config(text=f"Status: {result.status_message}", fg="green" if result.is_valid else "red")
            self.quality_label.config(text=f"Calitate: {result.quality_level}")
            defects_text = f"Defecte: {len(result.defects)}" if result.defects else "Defecte: Niciunul"
            self.defects_label.config(text=defects_text)

            self.root.after(self.delay, self.update_frame)
        except Exception as e:
            print(f"EROARE LIVE: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(100, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = TireQCViewer(root)
    root.mainloop()
