import cv2
import numpy as np
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import time

MM_TO_PX = 3.2

class DefectType(Enum):
    COLOR_MISSING = "culoare_lipsa"
    COLOR_WRONG = "culoare_gresita"
    LINE_BROKEN = "linie_intrerupta"
    WIDTH_WRONG = "latime_gresita"
    SPACING_WRONG = "spatiere_gresita"
    CONTAMINATION = "contaminare"
    EDGE_DEFECT = "defect_margine"
    LINE_SHIFTED = "linie_deplasata"

@dataclass
class Pattern:
    name: str
    colors: List[str]
    color_ranges: Dict[str, List[Tuple[List[int], List[int]]]]
    expected_widths: List[int] 
    expected_spacing: List[int]
    expected_positions_mm: Dict[str, int]
    expected_positions_px: Dict[str, int]
    tolerance_width: float = 0.15
    tolerance_spacing: float = 0.20
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
                "white": [
                    ([0, 0, 170], [180, 45, 255]) 
                ],
                "yellow": [
                    ([18, 20, 140], [42, 255, 255])
                ],
                "aqua": [
                    ([102, 101, 210], [108, 150, 255])
                ]
            },

            expected_widths=[30, 30, 30, 30],

            expected_spacing=[
                int(5 * MM_TO_PX),
                int(50 * MM_TO_PX),
                int(10 * MM_TO_PX)
            ],

            expected_positions_mm={
                "green": 80,
                "white": 70,
                "yellow": 20,
                "aqua": 15
            },

            expected_positions_px={
                c: int(mm * MM_TO_PX)
                for c, mm in {
                    "green": 80,
                    "white": 70,
                    "yellow": 20,
                    "aqua": 15
                }.items()
            },

            tolerance_width=0.5,       
            tolerance_spacing=0.5,      
            min_line_continuity=0.75      
        )
        
        self.patterns["YAWG"] = yawg_pattern
        self.line_shift_tolerance_ratio = 0.4
    
    def measure_actual_positions(self, image_path: str):
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        center_x = width // 2
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_stats = self._calculate_image_statistics(image)
        
        print(f"\n{'='*60}")
        print(f"Centru bandă: {center_x}px ({center_x/MM_TO_PX:.1f}mm)")
        print(f"{'='*60}")
        
        for color in ["green", "white", "yellow", "aqua"]:
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
            adapted_lower = np.array(lower, dtype=np.uint8)
            adapted_upper = np.array(upper, dtype=np.uint8)
            
            if self.adaptive_thresholds:
                if brightness_factor < 0.7:  
                    adapted_lower[1] = max(20, adapted_lower[1] - 20)  
                    adapted_lower[2] = max(30, adapted_lower[2] - 20)  
                elif brightness_factor > 1.3: 
                    adapted_upper[2] = min(255, adapted_upper[2] + 20)

            
            mask = cv2.inRange(hsv_image, adapted_lower, adapted_upper)
            full_mask = mask if full_mask is None else cv2.bitwise_or(full_mask, mask)
        
        return full_mask
    
    def _analyze_line_continuity(self, mask: np.ndarray, expected_height_ratio: float = 0.8) -> Dict:
        height, width = mask.shape
        
        if width == 0:
            return {
                'avg_continuity': 0,
                'min_continuity': 0,
                'interruptions': [],
                'continuity_scores': []
            }
        
        continuity_scores = []
        
        for col in range(width):
            column = mask[:, col]
            non_zero_pixels = np.count_nonzero(column)
            continuity = non_zero_pixels / height if height > 0 else 0
            continuity_scores.append(continuity)
        
        avg_continuity = np.mean(continuity_scores) if continuity_scores else 0
        min_continuity = np.min(continuity_scores) if continuity_scores else 0
        
        interruptions = []
        for i, score in enumerate(continuity_scores):
            if score < 0.5:  
                interruptions.append(i)
        
        return {
            'avg_continuity': avg_continuity,
            'min_continuity': min_continuity,
            'interruptions': interruptions,
            'continuity_scores': continuity_scores
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
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            if area > 100:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box)
                
                expected_perimeter = 2 * (rect[1][0] + rect[1][1])
                roughness = perimeter / expected_perimeter if expected_perimeter > 0 else 1
                
                if roughness > 1.3:  
                    center = (int(rect[0][0]), int(rect[0][1]))
                    defect = DefectReport(
                        defect_type=DefectType.EDGE_DEFECT,
                        severity=min((roughness - 1.0) / 0.5, 1.0),
                        position=center,
                        description=f"Margini neregulate (rugozitate {roughness:.2f})",
                        confidence=0.8
                    )
                    defects.append(defect)
        
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
        all_defects: List[DefectReport] = {}
        all_defects = []

        for i, color_name in enumerate(self.current_pattern.colors):
            color_ranges = self.current_pattern.color_ranges[color_name]

            mask = self._adaptive_color_detection(hsv, color_ranges, image_stats)
            line_info = self._detect_advanced_line(mask, color_name, i)

            if not line_info:
                all_defects.append(
                    DefectReport(
                        defect_type=DefectType.COLOR_MISSING,
                        severity=1.0,
                        position=(width // 2, height // 2),
                        description=f"Linie lipsă: {color_name}",
                        confidence=0.95
                    )
                )
                continue

            detected_lines[color_name] = line_info

            expected_width = self.current_pattern.expected_widths[i]
            width_tolerance = expected_width * self.current_pattern.tolerance_width

            if abs(line_info["width"] - expected_width) > width_tolerance:
                severity = abs(line_info["width"] - expected_width) / expected_width
                all_defects.append(
                    DefectReport(
                        defect_type=DefectType.WIDTH_WRONG,
                        severity=min(severity, 1.0),
                        position=(line_info["x_position"], height // 2),
                        description=(
                            f"Lățime incorectă la {color_name}: "
                            f"{line_info['width']}px (așteptat {expected_width}±{width_tolerance:.1f})"
                        ),
                        confidence=0.9
                    )
                )

            x, y, w, h = line_info["bounding_box"]
            line_mask = mask[y:y + h, x:x + w]
            continuity = self._analyze_line_continuity(line_mask)

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

    def _analyze_frame_absolute(self, image: np.ndarray):
            defects = []
            debug_info = {}

            image_stats = self._calculate_image_statistics(image)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            height, width = image.shape[:2]
            tire_center_x = width // 2

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

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest)
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

                if not found or coverage < 0.4:
                    defects.append(
                        DefectReport(
                            defect_type=DefectType.COLOR_MISSING,
                            severity=1.0,
                            position=(x_center_expected, height // 2),
                            description=f"Culoare {color} lipsă la poziția așteptată",
                            confidence=0.95
                        )
                    )
                    continue

                measured_offset = abs(current_center_abs[0] - tire_center_x)
                delta_px = abs(measured_offset - expected_px)
                tolerance_px = int(6 * MM_TO_PX)

                if delta_px > tolerance_px:
                    defects.append(
                        DefectReport(
                            defect_type=DefectType.LINE_SHIFTED,
                            severity=min(delta_px / expected_px, 1.0),
                            position=current_center_abs,
                            description=(
                                f"{color}: {measured_offset/MM_TO_PX:.1f}mm "
                                f"(așteptat {expected_px/MM_TO_PX:.1f}±6mm)"
                            ),
                            confidence=0.95
                        )
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
    
        required_order = ["aqua", "yellow", "white", "green"]
    
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
                defect_type=DefectType.SPACING_WRONG, 
                severity=1.0,
                position=(ordered[1][1]["x_position"], ordered[1][1]["y_position"]),
                description=f"Ordine gresita: {detected_order} != {required_order}",
                confidence=0.95
            )
        
        return None

    
    def _check_spacing(self, detected_lines: Dict[str, Dict]) -> List[DefectReport]:
        
        defects = []
    
        ordered = sorted(
            detected_lines.items(),
            key=lambda x: x[1]["bounding_box"][0]
        )
        
        edges = []
        for color, info in ordered:
            x, y, w, h = info["bounding_box"]
            edges.append((color, x, x + w))
        
        expected = self.current_pattern.expected_spacing
        
        for i in range(len(edges) - 1):
            color_prev, _, right_edge_prev = edges[i]
            color_next, left_edge_next, _ = edges[i + 1]
            
            actual_gap = left_edge_next - right_edge_prev
            expected_gap = expected[i]
            tolerance = expected_gap * self.current_pattern.tolerance_spacing
            
            print(f"[SPACING] {color_prev}->{color_next}: gap={actual_gap}px, "
                f"expected={expected_gap}±{tolerance:.1f}px")
            
            if abs(actual_gap - expected_gap) > max(tolerance, 8):  
                severity = min(abs(actual_gap - expected_gap) / expected_gap, 1.0)
                
                defect_x = (right_edge_prev + left_edge_next) // 2
                
                defects.append(
                    DefectReport(
                        defect_type=DefectType.SPACING_WRONG,
                        severity=severity,
                        position=(defect_x, 100),
                        description=(
                            f"Spatiere {color_prev}->{color_next}: {actual_gap}px "
                            f"(asteptat {expected_gap}±{tolerance:.1f}px, "
                            f"diferenta {abs(actual_gap - expected_gap):.1f}px)"
                        ),
                        confidence=0.90
                    )
                )
        
        return defects

    
    def _generate_status_messages(self, found_colors: Dict[str, bool], defects: List[DefectReport]) -> tuple:
        
        missing_colors = [color for color, found in found_colors.items() if not found]
        
        critical_defects = [d for d in defects if d.severity > 0.7]
        moderate_defects = [d for d in defects if 0.3 < d.severity <= 0.7]
        minor_defects = [d for d in defects if d.severity <= 0.3]
        
        if missing_colors:
            is_valid = False
            quality_level = "INACCEPTABIL"
            status_message = f"RESPINS - Lipsesc culori: {', '.join(missing_colors).upper()}"
            summary = f"Cauciucul NU poate fi folosit. Lipsesc {len(missing_colors)} culoare/culori."
            
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
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
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
                print_debug = (analyzed_frames % 30 == 0)

                if roi:
                    y1, y2, x1, x2 = roi
                    frame_roi = frame[y1:y2, x1:x2]
                else:
                    frame_roi = frame
                    x1, y1 = 0, 0

                debug_info = {}

                defects, debug_info = self._analyze_frame_absolute(frame_roi)

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
                    self._draw_debug_overlay(overlay, debug_info, roi_offset=(x1, y1))

                    for defect in defects:
                        if defect.defect_type != DefectType.LINE_SHIFTED:
                            continue

                        draw_x = x1 + defect.position[0]
                        draw_y = y1 + defect.position[1]

                        cv2.circle(
                            overlay,
                            (draw_x, draw_y),
                            10,
                            (0, 0, 255),
                            2
                        )

                    out.write(overlay)

            cap.release()
            if out:
                out.release()

            return {
                "total_frames": frame_index,
                "analyzed_frames": analyzed_frames,
                "summary": f"Procesat {analyzed_frames}/{frame_index} frame-uri."
            }
