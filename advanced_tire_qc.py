import cv2
import numpy as np
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import time
from collections import deque

MM_TO_PX = 3.2

class DefectType(Enum):
    COLOR_MISSING = "culoare_lipsa"
    COLOR_WRONG = "culoare_gresita"
    LINE_BROKEN = "linie_intrerupta"
    WIDTH_WRONG = "latime_gresita"
    CONTAMINATION = "contaminare"
    EDGE_DEFECT = "defect_margine"
    LINE_SHIFTED = "linie_deplasata"

@dataclass
class Pattern:
    name: str
    colors: List[str]
    color_ranges: Dict[str, List[Tuple[List[int], List[int]]]]
    expected_widths: List[int]
    expected_positions_mm: Dict[str, int]
    expected_positions_px: Dict[str, int]
    tolerance_width: float = 0.15
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
    status_message: str
    quality_level: str
    defects: List[DefectReport]
    detected_lines: Dict[str, Dict]
    processing_time: float
    summary: str

class AdvancedTireQualityChecker:
    def __init__(self, config_file: str = None):
        self.patterns = {}
        self.current_pattern = None
        self.debug_mode = False
        self.adaptive_thresholds = True
        self.line_shift_tolerance_ratio = 0.25
        self.debug_visual = True
        self.last_positions = {}
        self.shift_persistence = {}
        self.position_history = {
                "green": deque(maxlen=12),
                "white": deque(maxlen=12),
                "yellow": deque(maxlen=12),
                "aqua": deque(maxlen=12)
            }
        self.fixed_tire_center_x = None
        
        self._load_default_patterns()
        
        if config_file and os.path.exists(config_file):
            self._load_config(config_file)
    
    def _load_default_patterns(self):
        yawg_pattern = Pattern(
            name="YAWG",
            colors=["aqua", "yellow", "white", "green"],
            color_ranges={
                "green": [
                    ([65, 25, 130], [90, 255, 255]) 
                ],
                # "white": [
                #     ([0, 0, 230], [180, 12, 255]) 
                # ],
                "white": [
                    ([0, 0, 170], [180, 20 , 255]) 
                ],
                "yellow": [
                    ([18, 20, 140], [42, 255, 255])
                ],
                "aqua": [
                    ([102, 101, 210], [108, 150, 255])
                ]
            },

            expected_widths=[4, 6, 4, 6],

            expected_positions_mm={
                "green": 63,
                "white": 56,
                "yellow": 32,
                "aqua": 28
            },

            expected_positions_px={
                c: int(mm * MM_TO_PX)
                for c, mm in {
                    "green": 202,
                    "white": 178,
                    "yellow": 104,
                    "aqua": 90
                }.items()
            },

            tolerance_width=0.12,          
            min_line_continuity=0.43      
        )
        
        self.patterns["YAWG"] = yawg_pattern
        self.line_shift_tolerance_ratio = 0.4


    def _measure_effective_width(self, mask: np.ndarray) -> float:
        """Estimate the effective band thickness from a binary mask.

        Uses the median span of non-zero pixels per row (robust to jagged edges/outliers).
        Returns 0.0 if the mask is empty.
        """
        h, w = mask.shape[:2]
        if h == 0 or w == 0:
            return 0.0

        spans = []
        for yy in range(h):
            xs = np.where(mask[yy, :] > 0)[0]
            if xs.size > 0:
                spans.append(int(xs[-1] - xs[0] + 1))

        if not spans:
            return 0.0
        return float(np.median(spans))
    
        
    def measure_actual_positions(self, image_path: str):
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        center_x = width // 2
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_stats = self._calculate_image_statistics(image)
        
        print(f"\n{'='*60}")
        print(f"Centru bandă: {center_x}px ({center_x/MM_TO_PX:.1f}mm)")
        print(f"{'='*60}")
        
        for color in self.current_pattern.colors:
            ranges = self.current_pattern.color_ranges[color]
            mask = self._adaptive_color_detection(hsv, ranges, image_stats)
            
            cv2.imwrite(f"debug_mask_{color}.png", mask)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            print(f"\n{color} - Total contururi: {len(contours)}")
            
            if contours:
                large_contours = [c for c in contours if cv2.contourArea(c) > 100]
                print(f"  Contururi mari (>100px²): {len(large_contours)}")
                
                for i, contour in enumerate(large_contours[:3]):  
                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)
                    center_line_x = x + w // 2
                    distance_from_center_px = abs(center_x - center_line_x)
                    distance_from_center_mm = distance_from_center_px / MM_TO_PX
                    
                    print(f"  [{i}] x={center_line_x:4}px | area={area:5.0f}px² | "
                        f"dist: {distance_from_center_px:4}px ({distance_from_center_mm:.1f}mm)")
                
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                center_line_x = x + w // 2
                distance_from_center_px = abs(center_x - center_line_x)
                distance_from_center_mm = distance_from_center_px / MM_TO_PX
                
                print(f"  [SELECTAT] x={center_line_x:4}px | "
                    f"dist: {distance_from_center_px:4}px ({distance_from_center_mm:.1f}mm)")
            else:
                print(f"  NU DETECTAT")
        
        print(f"{'='*60}\n")

    def _draw_debug_overlay(self, overlay, debug_info, roi_offset=(0, 0)):
        off_x, off_y = roi_offset

        color_bgr = {
            "green": (0, 255, 0),
            "white": (255, 255, 255),
            "yellow": (0, 255, 255),
            "aqua": (255, 255, 0)  
        }

        for color, info in debug_info.items():
            if "roi" not in info:
                continue

            x1, x2, y1, y2 = info["roi"]
            x1 += off_x; x2 += off_x
            y1 += off_y; y2 += off_y

            mask = info.get("mask", None)
            if mask is not None:
                h, w = mask.shape[:2]
                if h > 0 and w > 0:
                    colored = np.zeros((h, w, 3), dtype=np.uint8)
                    bgr = color_bgr.get(color, (0, 0, 255))
                    colored[mask > 0] = bgr

                    roi_area = overlay[y1:y2, x1:x2]
                    if roi_area.shape[:2] == colored.shape[:2]:
                        overlay[y1:y2, x1:x2] = cv2.addWeighted(roi_area, 1.0, colored, 0.35, 0)

            bbox = info.get("bbox", None)
            if bbox:
                bx, by, bw, bh = bbox  
                cv2.rectangle(overlay, (x1 + bx, y1 + by), (x1 + bx + bw, y1 + by + bh), (255, 0, 0), 2)

            ref_center = info.get("ref_center", None)
            if ref_center:
                rx, ry = ref_center
                cv2.circle(overlay, (off_x + rx, off_y + ry), 6, (0, 255, 0), -1) 

            cur_center = info.get("current_center", None)
            if cur_center:
                cx, cy = cur_center
                cv2.circle(overlay, (off_x + cx, off_y + cy), 6, (0, 0, 255), -1) 

            cov = info.get("coverage", None)
            if cov is not None:
                cv2.putText(
                    overlay,
                    f"{color} cov={cov:.2f}",
                    (x1 + 5, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

    
    def add_pattern(self, pattern: Pattern):
        self.patterns[pattern.name] = pattern
    
    def set_current_pattern(self, pattern_name: str):
        if pattern_name in self.patterns:
            self.current_pattern = self.patterns[pattern_name]
        else:
            raise ValueError(f"Pattern {pattern_name} nu există")
    
    
    def _adaptive_color_detection(self, hsv_image: np.ndarray, color_ranges: List, image_stats: Dict) -> np.ndarray:
        full_mask = None
        
        brightness_factor = image_stats['mean_brightness'] / 128.0
        
        for lower, upper in color_ranges:
            adapted_lower = np.array(lower, dtype=np.uint16)
            adapted_upper = np.array(upper, dtype=np.uint16)

            if self.adaptive_thresholds:
                if brightness_factor < 0.7:  
                    adapted_lower[1] = max(20, adapted_lower[1] - 20)  
                    adapted_lower[2] = max(30, adapted_lower[2] - 20)  
                elif brightness_factor > 1.3: 
                    adapted_upper[2] = min(255, adapted_upper[2] + 20)

            adapted_lower = np.clip(adapted_lower, 0, 255).astype(np.uint8)
            adapted_upper = np.clip(adapted_upper, 0, 255).astype(np.uint8)
            
            mask = cv2.inRange(hsv_image, adapted_lower, adapted_upper)
            full_mask = mask if full_mask is None else cv2.bitwise_or(full_mask, mask)
        
        return full_mask
    
    def _analyze_line_continuity(self, mask: np.ndarray) -> Dict:
        height, width = mask.shape

        if width == 0 or height == 0:
            return {
                "avg_continuity": 0.0,
                "min_continuity": 0.0,
                "broken_ratio": 1.0,
                "continuity_scores": []
            }

        continuity_scores = []

        for col in range(width):
            column = mask[:, col]
            continuity_scores.append(np.count_nonzero(column) / height)

        continuity_scores = np.array(continuity_scores)

        avg_continuity = float(np.mean(continuity_scores))
        min_continuity = float(np.min(continuity_scores))

        broken_cols = continuity_scores < 0.5  # Schimbat de la 0.35 la 0.5 (mai strict)
        broken_ratio = float(np.sum(broken_cols) / width)

        return {
            "avg_continuity": avg_continuity,
            "min_continuity": min_continuity,
            "broken_ratio": broken_ratio,
            "continuity_scores": continuity_scores.tolist()
        }

    
    def _detect_contamination(self, image: np.ndarray, mask: np.ndarray) -> List[DefectReport]:
        defects = []
        
        background_mask = cv2.bitwise_not(mask)
        
        contours, _ = cv2.findContours(background_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 200: 
                x, y, w, h = cv2.boundingRect(contour)
                
                defect = DefectReport(
                    defect_type=DefectType.CONTAMINATION,
                    severity=min(area / 200.0, 1.0),
                    position=(x + w//2, y + h//2),
                    description=f"Posibilă contaminare (zona {area:.0f}px²)",
                    confidence=0.7
                )
                defects.append(defect)
        
        return defects
    
    def _analyze_line_edges(self, mask: np.ndarray) -> List[DefectReport]:
        defects = []

        h, w = mask.shape[:2]
        if h == 0 or w == 0:
            return defects

        # Reduce false positives from threshold noise by smoothing the binary mask a bit.
        try:
            proc = cv2.medianBlur(mask, 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel, iterations=1)
        except Exception:
            proc = mask

        contours, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return defects

        # Pick dominant band contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 100:
            return defects

        # Compare perimeter to ideal rectangle perimeter
        x, y, bw, bh = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, True)
        rect_perimeter = 2.0 * (bw + bh)
        peri_ratio = perimeter / (rect_perimeter + 1e-6)

        # Edge jitter along rows: left/right boundary variation
        roi = proc[y:y + bh, x:x + bw]
        left_xs = []
        right_xs = []
        rows = []
        for yy in range(bh):
            row = roi[yy, :]
            xs = np.where(row > 0)[0]
            if xs.size > 0:
                left_xs.append(x + int(xs[0]))
                right_xs.append(x + int(xs[-1]))
                rows.append(y + yy)

        if len(rows) < max(10, int(0.1 * bh)):
            return defects

        left_xs = np.array(left_xs)
        right_xs = np.array(right_xs)
        rows = np.array(rows)

        def jitter_metrics(edge_series: np.ndarray):
            if edge_series.size < 3:
                return 0.0, 0.0
            grad = np.diff(edge_series)
            mean_abs = float(np.mean(np.abs(grad)))
            high_frac = float(np.sum(np.abs(grad) > 1.5) / max(1, grad.size))
            return mean_abs, high_frac

        mean_left, frac_left = jitter_metrics(left_xs)
        mean_right, frac_right = jitter_metrics(right_xs)
        rough_score = max(mean_left, mean_right)
        high_frac = max(frac_left, frac_right)

        # Stricter thresholds to avoid false positives on good tires.
        PERI_THRESH = 1.10
        ROUGH_THRESH = 1.5
        HIGH_FRAC_THRESH = 0.35

        peri_flag = peri_ratio > PERI_THRESH
        jitter_flag = (rough_score > ROUGH_THRESH and high_frac > HIGH_FRAC_THRESH)

        # Require BOTH perimeter increase and jitter to flag a real edge defect.
        if peri_flag and jitter_flag:
            side = 'left' if mean_left >= mean_right else 'right'
            grads = np.diff(left_xs) if side == 'left' else np.diff(right_xs)
            if grads.size > 0:
                idx = int(np.argmax(np.abs(grads)))
                yy = rows[idx]
                xx = (left_xs[idx] if side == 'left' else right_xs[idx])
            else:
                yy = y + bh // 2
                xx = x + bw // 2

            # Map to severity with a moderate floor, but only when BOTH signals agree.
            severity_raw = (
                max(peri_ratio - PERI_THRESH, 0.0) / 0.15 * 0.5 +
                max(rough_score - ROUGH_THRESH, 0.0) / 2.0 * 0.5
            )
            severity = float(min(max(severity_raw, 0.35), 1.0))
            confidence = 0.9

            defects.append(
                DefectReport(
                    defect_type=DefectType.EDGE_DEFECT,
                    severity=severity,
                    position=(int(xx), int(yy)),
                    description=f"Margine neregulată {side}: peri {peri_ratio:.2f}, jitter {rough_score:.2f}",
                    confidence=confidence
                )
            )

        return defects
    
    def _calculate_image_statistics(self, image: np.ndarray) -> Dict:
        """Calculează statistici despre imagine pentru calibrare adaptivă"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return {
            'mean_brightness': np.mean(gray),
            'std_brightness': np.std(gray),
            'contrast': np.std(gray),
            'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var()
        }
    
    def _detect_wrong_color(self, hsv: np.ndarray, expected_x: int, height: int, width: int, expected_color: str) -> Optional[Tuple[str, Tuple[int, int]]]:
        """Detectează dacă la poziția așteptată există o linie cu culoare greșită"""
        roi_half = 50 
        
        x1 = max(0, expected_x - roi_half)
        x2 = min(width, expected_x + roi_half)
        
        roi_hsv = hsv[:, x1:x2]
        if roi_hsv.size == 0:
            return None
        
        bgr_roi = cv2.cvtColor(roi_hsv, cv2.COLOR_HSV2BGR)
        gray_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > height * 0.4 and w < 30:
                line_roi = roi_hsv[y:y+h, x:x+w]
                if line_roi.size == 0:
                    continue
                
                mean_hsv = cv2.mean(line_roi)
                h_val, s_val, v_val = mean_hsv[:3]
                
                best_match = None
                best_confidence = 0
                
                for color_name, ranges in self.current_pattern.color_ranges.items():
                    for lower, upper in ranges:
                        if (lower[0] <= h_val <= upper[0] and 
                            lower[1] <= s_val <= upper[1] and 
                            lower[2] <= v_val <= upper[2]):

                            if color_name != expected_color:
                                best_match = color_name
                                best_confidence = 1.0 
                                break
                    if best_match:
                        break
                
                if best_match:
                    position = (x1 + x + w // 2, height // 2)
                    return best_match, position
        
        return None
    
    def analyze_tire(self, image_path: str) -> QualityResult:
        start_time = time.time()
        image = cv2.imread(image_path)
        if image is None:
            return QualityResult(
                is_valid=False,
                status_message="EROARE LA ÎNCĂRCARE IMAGINE",
                quality_level="EROARE",
                defects=[],
                detected_lines={},
                processing_time=0.0,
                summary="Nu s-a putut citi imaginea."
            )

        image_stats = self._calculate_image_statistics(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = image.shape[:2]

        detected_lines: Dict[str, Dict] = {}
        all_defects: List[DefectReport] = []
        missing_colors = []

        for i, color_name in enumerate(self.current_pattern.colors):
            color_ranges = self.current_pattern.color_ranges[color_name]

            mask = self._adaptive_color_detection(hsv, color_ranges, image_stats)
            line_info = self._detect_advanced_line(mask, color_name, i)

            if not line_info:
                missing_colors.append(color_name)
                continue

            detected_lines[color_name] = line_info

            x, y, w, h = line_info["bounding_box"]
            line_mask = mask[y:y + h, x:x + w]

            expected_width = self.current_pattern.expected_widths[i]
            width_tolerance = expected_width * self.current_pattern.tolerance_width

            measured_width = self._measure_effective_width(line_mask)
            if measured_width <= 0:
                measured_width = float(line_info["width"])

            if abs(measured_width - expected_width) > width_tolerance:
                severity = abs(measured_width - expected_width) / max(expected_width, 1)
                all_defects.append(
                    DefectReport(
                        defect_type=DefectType.WIDTH_WRONG,
                        severity=min(severity, 1.0),
                        position=(line_info["x_position"], height // 2),
                        description=(
                            f"Lățime incorectă la {color_name}: "
                            f"{measured_width:.1f}px (așteptat {expected_width}±{width_tolerance:.1f})"
                        ),
                        confidence=0.9
                    )
                )
            continuity = self._analyze_line_continuity(line_mask)
            if (
                continuity["min_continuity"] < 0.5 or  # Schimbat de la 0.4 la 0.5 (mai strict)
                continuity["broken_ratio"] > 0.10  # Schimbat de la 0.15 la 0.10 (mai strict)
            ):
                all_defects.append(
                    DefectReport(
                        defect_type=DefectType.LINE_BROKEN,
                        severity=1.0,
                        position=(line_info["x_position"], height // 2),
                        description=(
                            f"Linie {color_name} întreruptă "
                            f"(min={continuity['min_continuity']:.2f}, "
                            f"broken={continuity['broken_ratio']*100:.0f}%)"
                        ),
                        confidence=0.95
                    )
                )
            
            # Verificare margini neregulate
            edge_defects = self._analyze_line_edges(line_mask)
            all_defects.extend(edge_defects)

        for color in missing_colors:
            wrong_color_info = self._detect_wrong_color(hsv, self.current_pattern.expected_positions_px[color], height, width, color)
            if wrong_color_info:
                wrong_color_name, position = wrong_color_info
                all_defects.append(
                    DefectReport(
                        defect_type=DefectType.COLOR_WRONG,
                        severity=1.0,
                        position=position,
                        description=f"Culoare greșită pentru {color}: detectată {wrong_color_name}",
                        confidence=0.85
                    )
                )
            else:
                all_defects.append(
                    DefectReport(
                        defect_type=DefectType.COLOR_MISSING,
                        severity=1.0,
                        position=(width // 2, height // 2),
                        description=f"Linie lipsă: {color}",
                        confidence=0.95
                    )
                )

        order_defect = self._check_exact_order(detected_lines)
        if order_defect:
            all_defects.append(order_defect)
        if len(detected_lines) >= 2:
            color_positions = {
                c: info["x_position"] for c, info in detected_lines.items()
            }

        status_message, quality_level, is_valid, summary = \
            self._generate_status_messages(
                {c: c in detected_lines for c in self.current_pattern.colors},
                all_defects
            )

        processing_time = time.time() - start_time

        return QualityResult(
            is_valid=is_valid,
            status_message=status_message,
            quality_level=quality_level,
            defects=all_defects,
            detected_lines=detected_lines,
            processing_time=processing_time,
            summary=summary
        )

    def _analyze_frame_absolute(self, image: np.ndarray, tire_center_x: int, x_offset: int = 0):
        defects = []
        debug_info = {}

        image_stats = self._calculate_image_statistics(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = image.shape[:2]
        if self.fixed_tire_center_x is None:
            raise RuntimeError("Centru bandă necalibrat")


        if not hasattr(self, "last_positions"):
            self.last_positions = {}
        if not hasattr(self, "shift_persistence"):
            self.shift_persistence = {}

        for color in self.current_pattern.colors:
            ranges = self.current_pattern.color_ranges[color]
            expected_px = self.current_pattern.expected_positions_px[color]

            roi_half_width = int(20 * MM_TO_PX)
            x_center_expected = tire_center_x - expected_px
            x1 = max(0, x_center_expected - roi_half_width)
            x2 = min(width, x_center_expected + roi_half_width)

            roi_hsv = hsv[:, x1:x2]
            if roi_hsv.size == 0:
                continue

            mask = self._adaptive_color_detection(roi_hsv, ranges, image_stats)

            coverage = np.count_nonzero(mask) / mask.size

            found = False
            current_center_abs = None
            bbox = None

            if coverage >= 0.15:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
            else:
                contours = []

            if contours:
                def score_contour(c):
                    x, y, w, h = cv2.boundingRect(c)
                    area = cv2.contourArea(c)
                    aspect = h / max(w, 1)

                    if area < 200:
                        return 0
                    if aspect < 3.0:   
                        return 0

                    return area * aspect

                best = max(contours, key=score_contour, default=None)
                if best is not None:
                    x, y, w, h = cv2.boundingRect(best)
                    current_center_abs = (x1 + x + w // 2, height // 2)
                    bbox = (x, y, w, h)
                    found = True

            debug_info[color] = {
                "mask": mask,
                "roi": (x1, x2, 0, height),
                "coverage": coverage,
                "current_center": current_center_abs,
                "expected_center": (x_center_expected, height // 2),
                "bbox": bbox,
                "found": found
            }

            if not found or current_center_abs is None or coverage < 0.4:
                continue

            prev = self.last_positions.get(color)
            if prev is not None:
                smoothed_center = int(0.7 * prev + 0.3 * current_center_abs[0])
            else:
                smoothed_center = current_center_abs[0]

            measured_offset_mm = abs(smoothed_center - tire_center_x) / MM_TO_PX
            expected_offset_mm = self.current_pattern.expected_positions_mm[color]
            delta_mm = abs(measured_offset_mm - expected_offset_mm)

            abs_offset_mm = abs(smoothed_center - tire_center_x) / MM_TO_PX
            abs_error_mm = abs(abs_offset_mm - expected_offset_mm)

            self.last_positions[color] = smoothed_center


            self.position_history[color].append(delta_mm)

            FAIL_MM = 7.0
            PERSISTENCE_FRAMES = 6

            bad = [d for d in self.position_history[color] if d > FAIL_MM]

            if len(bad) >= PERSISTENCE_FRAMES:
                defects.append(
                    DefectReport(
                        defect_type=DefectType.LINE_SHIFTED,
                        severity=min((delta_mm - FAIL_MM) / FAIL_MM, 1.0),
                        position=(smoothed_center - x_offset, height // 2),
                        description=(
                            f"{color} deviată: {measured_offset_mm:.1f}mm "
                            f"(așteptat {expected_offset_mm:.1f}±{FAIL_MM}mm)"
                        ),
                        confidence=0.95
                    )
                )

            if color == "white":
                print(
                    f"[DEBUG][WHITE] "
                    f"found={found} "
                    f"coverage={coverage:.2f} "
                    f"center={smoothed_center} "
                    f"expected={expected_px} "
                    f"dead_zone={int(3 * MM_TO_PX)} "
                    f"persist={self.shift_persistence.get(color, 0)}"
                )

        return defects, debug_info



    def _detect_advanced_line(self, mask: np.ndarray, color_name: str, color_index: int) -> Optional[Dict]:
       
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        height, width = mask.shape
        best_contour = None
        best_score = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            aspect_ratio = h / float(w) if w > 0 else 0
            height_ratio = h / height
            area_ratio = area / (w * h) if w * h > 0 else 0
            
            size_score = min(area / (height * 15), 1.0) 
            
            if aspect_ratio < 5:
                shape_score = aspect_ratio / 10.0  
            else:
                shape_score = min(aspect_ratio / 2.0, 1.0)
            
            height_score = height_ratio if height_ratio <= 1.0 else 0
            density_score = area_ratio
            
            total_score = (size_score * 0.35 +  
                        shape_score * 0.25 +  
                        height_score * 0.3 + 
                        density_score * 0.1)
            
            if (height_ratio > 0.4 and           
                aspect_ratio > 1.2 and            
                area > 150 and                   
                total_score > best_score):
                best_contour = contour
                best_score = total_score
        
        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            area = cv2.contourArea(best_contour)
            
            M = cv2.moments(best_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w//2, y + h//2
            
            return {
                'x_position': cx,
                'y_position': cy,
                'width': w,
                'height': h,
                'area': area,
                'bounding_box': (x, y, w, h),
                'confidence': best_score,
                'aspect_ratio': h / float(w) if w > 0 else 0,
                'height_ratio': h / height
            }
        
        return None
    
    def _check_exact_order(self, detected_lines: dict) -> Optional[DefectReport]:
    
        required_order = self.current_pattern.colors
    
        for color in required_order:
            if color not in detected_lines:
                return DefectReport(
                    defect_type=DefectType.COLOR_MISSING,
                    severity=1.0,
                    position=(0, 0),
                    description=f"Culoare lipsa: {color}",
                    confidence=0.95
                )
        
        ordered = sorted(
            detected_lines.items(),
            key=lambda x: x[1]["bounding_box"][0] 
        )
        
        detected_order = [color for color, _ in ordered]
        
        if detected_order != required_order:
            return DefectReport(
                defect_type=DefectType.LINE_SHIFTED, 
                severity=1.0,
                position=(ordered[1][1]["x_position"], ordered[1][1]["y_position"]),
                description=f"Ordine incorecta a culorilor: {'->'.join(detected_order)}",
                confidence=0.95
            )
        
        return None

    def _generate_status_messages(self, found_colors: Dict[str, bool], defects: List[DefectReport]) -> tuple:
        
        missing_colors = [color for color, found in found_colors.items() if not found]
        position_shifts = [d for d in defects if d.defect_type == DefectType.LINE_SHIFTED]
        wrong_color_defects = [d for d in defects if d.defect_type == DefectType.COLOR_WRONG]
        
        critical_defects = [d for d in defects if d.severity > 0.7]
        moderate_defects = [d for d in defects if 0.3 < d.severity <= 0.7]
        minor_defects = [d for d in defects if d.severity <= 0.3]
        
        if missing_colors or position_shifts or wrong_color_defects:
            is_valid = False
            quality_level = "INACCEPTABIL"
            reasons = []
            if missing_colors:
                reasons.append(f"lipsesc {', '.join(missing_colors).upper()}")
            if wrong_color_defects:
                wrong_desc = [d.description for d in wrong_color_defects]
                reasons.append(f"culori gresite: {'; '.join(wrong_desc)}")
            if position_shifts:
                reasons.append(f"deplasari pozitie ({len(position_shifts)})")
            status_message = f"RESPINS - {'; '.join(reasons)}"
            summary = f"Cauciucul NU poate fi folosit. Probleme: {'; '.join(reasons)}."
            
        elif len(critical_defects) > 0:
            is_valid = False
            quality_level = "INACCEPTABIL"
            defect_types = set([d.defect_type.value for d in critical_defects])
            status_message = f"RESPINS - Defecte critice: {len(critical_defects)}"
            summary = f"Cauciucul NU poate fi folosit. Defecte grave: {', '.join(defect_types)}."
            
        elif len(moderate_defects) > 2:
            is_valid = False
            quality_level = "ACCEPTABIL CU REZERVE"
            status_message = f"ATENȚIE - {len(moderate_defects)} defecte moderate detectate"
            summary = f"Cauciucul poate fi folosit cu aprobare specială. Necesită verificare suplimentară."
            
        elif len(moderate_defects) > 0:
            is_valid = True
            quality_level = "BUN"
            status_message = f"ACCEPTAT - Defecte minore: {len(moderate_defects)}"
            summary = f"Cauciucul este acceptabil. {len(moderate_defects)} defecte minore detectate."
            
        elif len(minor_defects) > 0:
            is_valid = True
            quality_level = "FOARTE BUN"
            status_message = "ACCEPTAT - Calitate foarte bună"
            summary = f"Cauciucul este de calitate foarte bună. Doar {len(minor_defects)} imperfecțiuni minore."
            
        else:
            is_valid = True
            quality_level = "EXCELENT"
            status_message = "ACCEPTAT - Pattern perfect!"
            summary = "Cauciucul este perfect! Niciun defect detectat."
        
        return status_message, quality_level, is_valid, summary
    
    def draw_detected_lines(self, image, detected_lines, offset=(0, 0)):
        ox, oy = offset
        color_map = {
            "green": (0, 255, 0),
            "white": (255, 255, 255),
            "yellow": (0, 255, 255),
            "aqua": (255, 255, 0)
        }

        for color, info in detected_lines.items():
            x, y, w, h = info["bounding_box"]
            cx = info["x_position"]
            cy = info["y_position"]

            cv2.rectangle(
                image,
                (ox + x, oy + y),
                (ox + x + w, oy + y + h),
                color_map.get(color, (0, 0, 255)),
                2
            )

            cv2.circle(
                image,
                (ox + cx, oy + cy),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                image,
                color,
                (ox + x, oy + y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_map.get(color, (0, 0, 255)),
                1
            )


    def analyze_tire_frame(self, frame: np.ndarray) -> QualityResult:
        start_time = time.time()

        image_stats = self._calculate_image_statistics(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]

        detected_lines = {}
        all_defects = []
        missing_colors = []

        for i, color_name in enumerate(self.current_pattern.colors):
            ranges = self.current_pattern.color_ranges[color_name]
            mask = self._adaptive_color_detection(hsv, ranges, image_stats)

            if self.debug_mode:
                try:
                    cv2.imwrite(f"debug_mask_frame_{color_name}.png", mask)
                except Exception:
                    pass

            line_info = self._detect_advanced_line(mask, color_name, i)

            if not line_info:
                missing_colors.append(color_name)
                continue

            detected_lines[color_name] = line_info

            x, y, w, h = line_info["bounding_box"]
            line_mask = mask[y:y + h, x:x + w]

            # Width validation (frame mode)
            expected_width = self.current_pattern.expected_widths[i]
            width_tolerance = expected_width * self.current_pattern.tolerance_width
            measured_width = self._measure_effective_width(line_mask)
            if measured_width <= 0:
                measured_width = float(line_info["width"])

            if abs(measured_width - expected_width) > width_tolerance:
                severity = min(abs(measured_width - expected_width) / max(expected_width, 1), 1.0)
                all_defects.append(
                    DefectReport(
                        defect_type=DefectType.WIDTH_WRONG,
                        severity=severity,
                        position=(line_info["x_position"], line_info["y_position"]),
                        description=(
                            f"Lățime incorectă {color_name}: {measured_width:.1f}px (așteptat {expected_width}±{width_tolerance:.1f})"
                        ),
                        confidence=0.9
                    )
                )
            continuity = self._analyze_line_continuity(line_mask)

            if continuity['avg_continuity'] < self.current_pattern.min_line_continuity:
                all_defects.append(
                    DefectReport(
                        defect_type=DefectType.LINE_BROKEN,
                        severity=1.0 - continuity['avg_continuity'],
                        position=(line_info["x_position"], line_info["y_position"]),
                        description=f"Linie întreruptă: {color_name} (continuitate {continuity['avg_continuity']:.2f})",
                        confidence=0.9
                    )
                )

            # Edge defects: detect irregular margins within the detected line
            edge_defects = self._analyze_line_edges(line_mask)
            for d in edge_defects:
                # Translate defect position from line_mask (local) to frame coordinates
                all_defects.append(
                    DefectReport(
                        defect_type=d.defect_type,
                        severity=d.severity,
                        position=(x + d.position[0], y + d.position[1]),
                        description=d.description,
                        confidence=d.confidence
                    )
                )

        for color in missing_colors:
            wrong_color_info = self._detect_wrong_color(hsv, self.current_pattern.expected_positions_px[color], height, width, color)
            if wrong_color_info:
                wrong_color_name, position = wrong_color_info
                all_defects.append(
                    DefectReport(
                        defect_type=DefectType.COLOR_WRONG,
                        severity=1.0,
                        position=position,
                        description=f"Culoare greșită pentru {color}: detectată {wrong_color_name}",
                        confidence=0.85
                    )
                )
            else:
                all_defects.append(
                    DefectReport(
                        defect_type=DefectType.COLOR_MISSING,
                        severity=1.0,
                        position=(width // 2, height // 2),
                        description=f"Linie lipsă: {color}",
                        confidence=0.95
                    )
                )

        status_message, quality_level, is_valid, summary = \
            self._generate_status_messages(
                {c: c in detected_lines for c in self.current_pattern.colors},
                all_defects
            )

        return QualityResult(
            is_valid=is_valid,
            status_message=status_message,
            quality_level=quality_level,
            defects=all_defects,
            detected_lines=detected_lines,
            processing_time=time.time() - start_time,
            summary=summary
        )

    
    def save_debug_image(self, image_path: str, result: QualityResult, output_path: str):
        image = cv2.imread(image_path)
        if image is None:
            return
        
        for defect in result.defects:
            color = (0, 0, 255)  
            if defect.severity < 0.3:
                color = (0, 255, 255)  
            elif defect.severity < 0.7:
                color = (0, 165, 255) 
            
            cv2.circle(image, defect.position, 10, color, 2)
            cv2.putText(image, defect.defect_type.value, 
                       (defect.position[0] + 15, defect.position[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        status_color = (0, 255, 0) if result.is_valid else (0, 0, 255)
        
        cv2.putText(image, result.quality_level, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, result.status_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        
        cv2.imwrite(output_path, image)

    def analyze_video(
            self,
            video_path: str,
            output_video_path: str = None,
            roi: tuple = None,
            frame_skip: int = 1,
            stop_after: int = None
        ):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Nu pot deschide videoclipul: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if output_video_path:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                out = cv2.VideoWriter(
                    output_video_path,
                    fourcc,
                    fps,
                    (frame_width, frame_height)
                )
            else:
                out = None

            frame_index = 0
            analyzed_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1
                if frame_index % frame_skip != 0:
                    continue

                if stop_after and analyzed_frames >= stop_after:
                    break

                analyzed_frames += 1

                if roi:
                    y1, y2, x1, x2 = roi
                    frame_roi = frame[y1:y2, x1:x2]
                else:
                    frame_roi = frame
                    x1, y1 = 0, 0
                    
                result = self.analyze_tire_frame(frame_roi)

                defects_abs, debug_info = self._analyze_frame_absolute(frame_roi, tire_center_x=self.fixed_tire_center_x-x1, x_offset=x1)

                for color, info in result.detected_lines.items():
                    abs_x = info["x_position"] + x1
                    measured_offset_mm = abs(abs_x - self.fixed_tire_center_x) / MM_TO_PX
                    expected_offset_mm = self.current_pattern.expected_positions_mm[color]
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
                    result.defects.append(
                        DefectReport(
                            defect_type=d.defect_type,
                            severity=d.severity,
                            position=d.position,  
                            description=d.description,
                            confidence=d.confidence
                        )
                    )

                status_message, quality_level, is_valid, summary = \
                    self._generate_status_messages(
                        {c: c in result.detected_lines for c in self.current_pattern.colors},
                        result.defects
                    )

                result.status_message = status_message
                result.quality_level = quality_level
                result.is_valid = is_valid
                result.summary = summary


                if out is not None:
                    overlay = frame.copy()

                    if roi:
                        cv2.rectangle(
                            overlay,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 255),
                            2
                        )

                    for color, info in result.detected_lines.items():
                        x, y, w, h = info["bounding_box"]
                        cx = info["x_position"]
                        cy = info["y_position"]

                        BOX_COLOR = (255, 0, 0)  
                        CENTER_COLOR = (0, 0, 255)

                        cv2.rectangle(
                            overlay,
                            (x1 + x, y1 + y),
                            (x1 + x + w, y1 + y + h),
                            BOX_COLOR,
                            2
                        )

                        cv2.circle(
                            overlay,
                            (x1 + cx, y1 + cy),
                            5,
                            CENTER_COLOR,
                            -1
                        )


                        cv2.putText(
                            overlay,
                            color,
                            (x1 + x, y1 + y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            BOX_COLOR,
                            1
                        )

                    cv2.line(
                        overlay,
                        (self.fixed_tire_center_x, 0),
                        (self.fixed_tire_center_x, frame.shape[0]),
                        (0, 255, 0),
                        2
                    )


                    for defect in result.defects:
                        dx = x1 + defect.position[0]
                        dy = y1 + defect.position[1]

                        if defect.severity > 0.7:
                            col = (0, 0, 255)
                        elif defect.severity > 0.3:
                            col = (0, 165, 255)
                        else:
                            col = (0, 255, 255)

                        cv2.circle(overlay, (dx, dy), 10, col, 2)

                    verdict_color = (0, 255, 0) if result.is_valid else (0, 0, 255)

                    cv2.putText(
                        overlay,
                        result.quality_level,
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        verdict_color,
                        2
                    )

                    cv2.putText(
                        overlay,
                        result.status_message,
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        verdict_color,
                        1
                    )

                    y = 90
                    for defect in result.defects:
                        txt = f"{defect.defect_type.value} sev={defect.severity:.2f}"
                        cv2.putText(
                            overlay,
                            txt,
                            (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 0, 255),
                            1
                        )
                        y += 18


                    out.write(overlay)

            cap.release()
            if out:
                out.release()

            return {
                "total_frames": frame_index,
                "analyzed_frames": analyzed_frames,
                "summary": f"Procesat {analyzed_frames}/{frame_index} frame-uri."
            }