import cv2
import numpy as np
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

MM_TO_PX = 3.2

# =========================================================
# ENUMS & DATA STRUCTURES
# =========================================================

class DefectType(Enum):
    COLOR_MISSING = "culoare_lipsa"
    LINE_SHIFTED = "linie_deplasata"
    WIDTH_WRONG = "latime_gresita"
    LINE_DEVIATED = "linie_deviata"
    LINE_BROKEN = "linie_intrerupta"


@dataclass
class Pattern:
    name: str
    colors: List[str]
    color_ranges: Dict[str, List[Tuple[List[int], List[int]]]]

    expected_widths: List[int]

    expected_positions_mm: Dict[str, float]
    expected_positions_px: Dict[str, int]

    tolerance_width: float = 0.5
    tolerance_position_mm: float = 6.0   # ±6 mm
    tolerance_straightness_mm: float = 2.0
    min_line_continuity: float = 0.85


@dataclass
class DefectReport:
    defect_type: DefectType
    severity: float
    position: Tuple[int, int]
    description: str
    confidence: float


@dataclass
class QualityResult:
    is_valid: bool
    defects: List[DefectReport]
    detected_lines: Dict[str, Dict]
    processing_time: float


# =========================================================
# MAIN CLASS
# =========================================================

class AdvancedTireQualityChecker:

    def __init__(self):
        self.patterns: Dict[str, Pattern] = {}
        self.current_pattern: Optional[Pattern] = None
        self._load_patterns()

    # -----------------------------------------------------

    def _load_patterns(self):
        expected_positions_mm = {
            "yellow": 15,
            "aqua": 40,
            "white": 70,
            "green": 80
        }

        expected_positions_px = {
            c: int(mm * MM_TO_PX)
            for c, mm in expected_positions_mm.items()
        }

        yawg = Pattern(
            name="YAWG",
            colors=["yellow", "aqua", "white", "green"],
            color_ranges={
                "green": [([65, 25, 130], [90, 255, 255])],
                "white": [([0, 0, 170], [180, 45, 255])],
                "yellow": [([18, 20, 140], [42, 255, 255])],
                "aqua": [([90, 40, 70], [130, 255, 255])]
            },
            expected_widths=[30, 30, 30, 30],
            expected_positions_mm=expected_positions_mm,
            expected_positions_px=expected_positions_px,
            tolerance_straightness_mm=2.5,
            min_line_continuity=0.85
        )

        self.patterns["YAWG"] = yawg
        self.current_pattern = yawg

    # =====================================================
    # CORE PIPELINE
    # =====================================================

    def analyze_image(self, image_path: str) -> QualityResult:
        start = time.time()
        image = cv2.imread(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h, w = image.shape[:2]
        tire_center_x = w // 2

        detected = {}
        defects = []

        for idx, color in enumerate(self.current_pattern.colors):
            mask = self._detect_color(hsv, color)
            line = self._detect_vertical_line(mask)

            if not line:
                defects.append(
                    DefectReport(
                        DefectType.COLOR_MISSING,
                        1.0,
                        (tire_center_x, h // 2),
                        f"Linie {color} lipsa",
                        0.95
                    )
                )
                continue

            detected[color] = line

            # ----------------------------
            # WIDTH CHECK
            # ----------------------------
            expected_w = self.current_pattern.expected_widths[idx]
            if abs(line["width"] - expected_w) > expected_w * self.current_pattern.tolerance_width:
                defects.append(
                    DefectReport(
                        DefectType.WIDTH_WRONG,
                        0.6,
                        (line["x"], h // 2),
                        f"{color}: latime {line['width']}px (asteptat {expected_w})",
                        0.9
                    )
                )

            # ----------------------------
            # STRAIGHTNESS CHECK
            # ----------------------------
            max_dev_mm = line["max_deviation_px"] / MM_TO_PX
            if max_dev_mm > self.current_pattern.tolerance_straightness_mm:
                defects.append(
                    DefectReport(
                        DefectType.LINE_DEVIATED,
                        min(max_dev_mm / self.current_pattern.tolerance_straightness_mm, 1.0),
                        (line["x"], h // 2),
                        f"{color}: deviere max {max_dev_mm:.2f}mm",
                        0.9
                    )
                )

            # ----------------------------
            # CONTINUITY CHECK
            # ----------------------------
            if line["continuity"] < self.current_pattern.min_line_continuity:
                defects.append(
                    DefectReport(
                        DefectType.LINE_BROKEN,
                        min(1.0 - line["continuity"], 1.0),
                        (line["x"], h // 2),
                        f"{color}: continuitate {line['continuity']*100:.1f}%",
                        0.9
                    )
                )

            # ----------------------------
            # POSITION CHECK (CORE)
            # ----------------------------
            measured_px = abs(line["x"] - tire_center_x)
            expected_px = self.current_pattern.expected_positions_px[color]
            delta_px = abs(measured_px - expected_px)
            tolerance_px = int(self.current_pattern.tolerance_position_mm * MM_TO_PX)

            if delta_px > tolerance_px:
                defects.append(
                    DefectReport(
                        DefectType.LINE_SHIFTED,
                        min(delta_px / expected_px, 1.0),
                        (line["x"], h // 2),
                        (
                            f"{color}: {measured_px/MM_TO_PX:.1f}mm "
                            f"(nominal {expected_px/MM_TO_PX:.1f}±{self.current_pattern.tolerance_position_mm}mm)"
                        ),
                        0.95
                    )
                )

        return QualityResult(
            is_valid=len(defects) == 0,
            defects=defects,
            detected_lines=detected,
            processing_time=time.time() - start
        )

    # =====================================================
    # DETECTION UTILITIES
    # =====================================================

    def _detect_color(self, hsv, color):
        mask_total = None
        for low, high in self.current_pattern.color_ranges[color]:
            m = cv2.inRange(hsv, np.array(low), np.array(high))
            mask_total = m if mask_total is None else cv2.bitwise_or(mask_total, m)
        return mask_total

    def _detect_vertical_line(self, mask) -> Optional[Dict]:
        """
        Detectie ROBUSTA pentru linii verticale fragmentate
        (PROIECTIE PE AXA X)
        """
        h, w = mask.shape
        col_sum = np.sum(mask > 0, axis=0)

        valid_cols = np.where(col_sum > h * 0.25)[0]
        if len(valid_cols) < 5:
            return None

        x_left = int(valid_cols[0])
        x_right = int(valid_cols[-1])
        cx = (x_left + x_right) // 2

        ys, xs = np.where(mask > 0)
        if len(xs) < 50:
            return None

        continuity_rows = np.unique(ys)
        continuity = len(continuity_rows) / float(h)

        vx, vy, x0, y0 = cv2.fitLine(
            np.column_stack((xs, ys)).astype(np.float32),
            cv2.DIST_L2,
            0,
            0.01,
            0.01
        )
        dx = xs - x0
        dy = ys - y0
        denom = np.hypot(vx, vy)
        distances = np.abs(vx * dy - vy * dx) / denom
        max_deviation_px = float(np.max(distances)) if len(distances) else 0.0

        return {
            "x": cx,
            "width": x_right - x_left + 1,
            "bounding_box": (x_left, 0, x_right - x_left + 1, h),
            "max_deviation_px": max_deviation_px,
            "continuity": continuity
        }


# =========================================================
# QUICK TEST
# =========================================================

if __name__ == "__main__":
    qc = AdvancedTireQualityChecker()
    result = qc.analyze_image("reference.jpg")

    print("VALID:", result.is_valid)
    for d in result.defects:
        print(d.description)
