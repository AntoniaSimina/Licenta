import cv2
import tkinter as tk
from PIL import Image, ImageTk
from advanced_tire_qc import AdvancedTireQualityChecker

VIDEO_PATH = r"C:\Users\Antonia\Desktop\Licenta 2.0\output_overlay.avi"

class TireQCViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Tire Quality Control")

        # -----------------------------
        # Init checker (doar pt info)
        # -----------------------------
        self.checker = AdvancedTireQualityChecker()
        self.checker.set_current_pattern("YAWG")
        pattern = self.checker.current_pattern

        # -----------------------------
        # Layout
        # -----------------------------
        self.video_label = tk.Label(root)
        self.video_label.grid(row=0, column=0, padx=10, pady=10)

        info_frame = tk.Frame(root)
        info_frame.grid(row=0, column=1, sticky="n", padx=10)

        # -----------------------------
        # Pattern info
        # -----------------------------
        tk.Label(
            info_frame,
            text=f"Pattern: {pattern.name}",
            font=("Arial", 16, "bold")
        ).pack(anchor="w", pady=(0, 10))

        tk.Label(
            info_frame,
            text="Culori detectate:",
            font=("Arial", 12, "bold")
        ).pack(anchor="w")

        color_map = {
            "green": "#00aa00",
            "white": "#dddddd",
            "yellow": "#ffff00",
            "aqua": "#00ffff"
        }

        for color in pattern.colors:
            row = tk.Frame(info_frame)
            row.pack(anchor="w", pady=3)

            box = tk.Canvas(row, width=20, height=20)
            box.pack(side="left", padx=(0, 6))
            box.create_rectangle(
                0, 0, 20, 20,
                fill=color_map.get(color, "gray")
            )

            tk.Label(
                row,
                text=f"{color.upper()}  –  {pattern.expected_positions_mm[color]} mm",
                font=("Arial", 11)
            ).pack(side="left")

        # -----------------------------
        # Video
        # -----------------------------
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.delay = int(1000 / self.fps)

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (900, 500))

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(self.delay, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = TireQCViewer(root)
    root.mainloop()
