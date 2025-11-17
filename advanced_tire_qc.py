import cv2
import numpy as np
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import time

class DefectType(Enum):
    COLOR_MISSING = "culoare_lipsa"
    COLOR_WRONG = "culoare_gresita"
    LINE_BROKEN = "linie_intrerupta"
    WIDTH_WRONG = "latime_gresita"
    SPACING_WRONG = "spatiere_gresita"
    CONTAMINATION = "contaminare"
    EDGE_DEFECT = "defect_margine"

@dataclass
class Pattern:
    name: str
    colors: List[str]
    color_ranges: Dict[str, List[Tuple[List[int], List[int]]]]
    expected_widths: List[int]  # în pixeli
    expected_spacing: List[int]  # distanțele între linii
    tolerance_width: float = 0.15  # 15% toleranță
    tolerance_spacing: float = 0.20  # 20% toleranță
    min_line_continuity: float = 0.85  # 85% din înălțime trebuie să fie continuă

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
    quality_level: str   # "EXCELENT", "BUN", "ACCEPTABIL", "INACCEPTABIL"
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
        
        self._load_default_patterns()
        
        if config_file and os.path.exists(config_file):
            self._load_config(config_file)
    
    def _load_default_patterns(self):
        """Încarcă pattern-urile implicite"""
        rgb_pattern = Pattern(
            name="RGB_Classic",
            colors=["red", "green", "blue"],
            color_ranges={
                "red": [([0, 50, 50], [15, 255, 255]), ([165, 50, 50], [180, 255, 255])],
                "green": [([35, 50, 50], [85, 255, 255])],
                "blue": [([95, 50, 50], [135, 255, 255])]
            },
            expected_widths=[40, 35, 40],  # lățimi așteptate
            # CORECTARE: spacing între CENTRE = lățime/2 + gap + lățime/2
            # Roșu(40) la Verde(35): 40/2 + 50 + 35/2 = 87.5
            # Verde(35) la Albastru(40): 35/2 + 50 + 40/2 = 87.5  
            expected_spacing=[87, 87],     # distanță între CENTRE
            tolerance_width=0.15,          # Toleranță mare pentru lățime (40%)
            tolerance_spacing=0.20,        # Toleranță mare pentru spațiere (40%)
            min_line_continuity=0.80       # Continuitate minimă 80%
        )
        self.patterns["RGB_Classic"] = rgb_pattern
        self.current_pattern = rgb_pattern
    
    def add_pattern(self, pattern: Pattern):
        """Adaugă un pattern nou"""
        self.patterns[pattern.name] = pattern
    
    def set_current_pattern(self, pattern_name: str):
        """Setează pattern-ul curent"""
        if pattern_name in self.patterns:
            self.current_pattern = self.patterns[pattern_name]
        else:
            raise ValueError(f"Pattern {pattern_name} nu există")
    
    def _adaptive_color_detection(self, hsv_image: np.ndarray, color_ranges: List, image_stats: Dict) -> np.ndarray:
        """Detecție adaptivă de culori bazată pe statisticile imaginii"""
        full_mask = None
        
      
        brightness_factor = image_stats['mean_brightness'] / 128.0
        
        for lower, upper in color_ranges:
           
            adapted_lower = np.array(lower, dtype=np.uint8)
            adapted_upper = np.array(upper, dtype=np.uint8)
            
            if self.adaptive_thresholds:
                if brightness_factor < 0.7:  # imagine întunecată
                    adapted_lower[1] = max(20, adapted_lower[1] - 20)  # relaxează saturation
                    adapted_lower[2] = max(30, adapted_lower[2] - 20)  # relaxează value
                elif brightness_factor > 1.3:  # imagine prea luminoasă
                    adapted_upper[2] = min(255, adapted_upper[2] + 20)
            
            mask = cv2.inRange(hsv_image, adapted_lower, adapted_upper)
            full_mask = mask if full_mask is None else cv2.bitwise_or(full_mask, mask)
        
        return full_mask
    
    def _analyze_line_continuity(self, mask: np.ndarray, expected_height_ratio: float = 0.8) -> Dict:
        """Analizează continuitatea unei linii - CORECTATĂ"""
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
            # Întrerupere = mai puțin de 50% din înălțime (nu 80%!)
            if score < 0.5:  
                interruptions.append(i)
        
        return {
            'avg_continuity': avg_continuity,
            'min_continuity': min_continuity,
            'interruptions': interruptions,
            'continuity_scores': continuity_scores
        }
    
    def _detect_contamination(self, image: np.ndarray, mask: np.ndarray) -> List[DefectReport]:
        """Detectează contaminarea (pete, murdărie)"""
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
        """Analizează marginile liniilor pentru defecte"""
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
        """Analiză completă a unui cauciuc"""
        start_time = time.time()
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Nu s-a putut încărca imaginea {image_path}")
        except Exception as e:
            return QualityResult(
                is_valid=False,
                status_message="EROARE LA ÎNCĂRCAREA IMAGINII",
                quality_level="EROARE",
                defects=[],
                detected_lines={},
                processing_time=time.time() - start_time,
                summary=f"Eroare: {str(e)}"
            )
        

        image_stats = self._calculate_image_statistics(image)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = image.shape[:2]
        
        detected_lines = {}
        all_defects = []
        found_colors = {}
        color_positions = {}
        
      
        for i, color_name in enumerate(self.current_pattern.colors):
            color_ranges = self.current_pattern.color_ranges[color_name]
            
            mask = self._adaptive_color_detection(hsv, color_ranges, image_stats)
            
            line_info = self._detect_advanced_line(mask, color_name, i)
            
            if line_info:
                detected_lines[color_name] = line_info
                found_colors[color_name] = True
                color_positions[color_name] = line_info['x_position']
                
                x, y, w, h = line_info['bounding_box']
                line_mask = mask[y:y+h, x:x+w]  # Extrage doar zona liniei
                continuity_analysis = self._analyze_line_continuity(line_mask)
                
                expected_width = self.current_pattern.expected_widths[i]
                width_tolerance = expected_width * self.current_pattern.tolerance_width
                
                if abs(line_info['width'] - expected_width) > width_tolerance:
                    severity = abs(line_info['width'] - expected_width) / expected_width
                    defect = DefectReport(
                        defect_type=DefectType.WIDTH_WRONG,
                        severity=min(severity, 1.0),
                        position=(line_info['x_position'], height//2),
                        description=f"Lățime incorectă: {line_info['width']}px (așteptat {expected_width}±{width_tolerance:.1f}px)",
                        confidence=0.9
                    )
                    all_defects.append(defect)
                
                if continuity_analysis['avg_continuity'] < self.current_pattern.min_line_continuity:
                    defect = DefectReport(
                        defect_type=DefectType.LINE_BROKEN,
                        severity=1.0 - continuity_analysis['avg_continuity'],
                        position=(line_info['x_position'], height//2),
                        description=f"Linie întreruptă (continuitate {continuity_analysis['avg_continuity']:.2f})",
                        confidence=0.8
                    )
                    all_defects.append(defect)
                
                edge_defects = self._analyze_line_edges(mask)
                contamination_defects = self._detect_contamination(image, mask)
                all_defects.extend(edge_defects)
                all_defects.extend(contamination_defects)
                
            else:
                found_colors[color_name] = False
                defect = DefectReport(
                    defect_type=DefectType.COLOR_MISSING,
                    severity=1.0,
                    position=(width//2, height//2),
                    description=f"Linie {color_name} lipsă sau prea mică",
                    confidence=0.9
                )
                all_defects.append(defect)
        
        if len(color_positions) >= 2:
            spacing_defects = self._check_spacing(color_positions)
            all_defects.extend(spacing_defects)
        
        
        status_message, quality_level, is_valid, summary = self._generate_status_messages(found_colors, all_defects)
        
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
    
    def _detect_advanced_line(self, mask: np.ndarray, color_name: str, color_index: int) -> Optional[Dict]:
        """Detecție avansată de linii cu informații detaliate - mai permisivă"""
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
            
        
            size_score = min(area / (height * 20), 1.0) 
            shape_score = min(aspect_ratio / 2.0, 1.0) if aspect_ratio <= 15 else 1.0 / (aspect_ratio / 15.0)  
            height_score = height_ratio if height_ratio <= 1.0 else 0
            density_score = area_ratio
            
        
            total_score = (size_score * 0.3 + shape_score * 0.3 + 
                          height_score * 0.3 + density_score * 0.1)
            
          
            if (height_ratio > 0.5 and aspect_ratio > 1.5 and 
                area > 200 and total_score > best_score):      
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
    
    def _check_spacing(self, color_positions: Dict[str, int]) -> List[DefectReport]:
        """Verifica spatierile intre linii"""
        defects = []
        
        if len(color_positions) < 2:
            return defects
        
        sorted_positions = sorted(color_positions.items(), key=lambda x: x[1])
        x_positions = [pos[1] for pos in sorted_positions]
        
        actual_distances = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
        expected_distances = self.current_pattern.expected_spacing
        
        for i, (actual, expected) in enumerate(zip(actual_distances, expected_distances)):
            tolerance = expected * self.current_pattern.tolerance_spacing
            min_acceptable = expected - tolerance
            max_acceptable = expected + tolerance
            
            if actual < min_acceptable or actual > max_acceptable:
                
                if actual < min_acceptable:
                    excess = min_acceptable - actual
                else:
                    excess = actual - max_acceptable
                
            
                severity = min(0.7 + (excess / expected) * 0.3, 1.0)
                
                defect_x = (x_positions[i] + x_positions[i+1]) // 2
                
                defect = DefectReport(
                    defect_type=DefectType.SPACING_WRONG,
                    severity=severity,
                    position=(defect_x, 100),
                    description=f"Spatiere incorecta: {actual}px (asteptat {expected}±{tolerance:.1f}px)",
                    confidence=0.85
                )
                defects.append(defect)
        
        return defects
    
    def _calculate_quality_score(self, found_colors: Dict[str, bool], defects: List[DefectReport]) -> float:
        """Calculează scorul de calitate (0-100) - mai permisiv pentru imagini generate"""
        base_score = 100.0
        
        
        missing_colors = sum(1 for found in found_colors.values() if not found)
        base_score -= missing_colors * 20.0  
        
        for defect in defects:
            penalty = defect.severity * defect.confidence
            
            if defect.defect_type == DefectType.COLOR_MISSING:
                penalty *= 15.0
            elif defect.defect_type == DefectType.LINE_BROKEN:
                penalty *= 10.0  
            elif defect.defect_type == DefectType.WIDTH_WRONG:
                penalty *= 6.0   
            elif defect.defect_type == DefectType.SPACING_WRONG:
                penalty *= 8.0   
            elif defect.defect_type == DefectType.CONTAMINATION:
                penalty *= 3.0   
            elif defect.defect_type == DefectType.EDGE_DEFECT:
                penalty *= 2.0   
            
            base_score -= penalty
        
        return max(0.0, min(100.0, base_score))
    
    def _generate_status_messages(self, found_colors: Dict[str, bool], defects: List[DefectReport]) -> tuple:
        """Generează mesaje clare despre status"""
        
      
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
        """Salvează o imagine cu defectele marcate"""
        image = cv2.imread(image_path)
        if image is None:
            return
       
        for defect in result.defects:
            color = (0, 0, 255)  # roșu pentru defecte
            if defect.severity < 0.3:
                color = (0, 255, 255)  # galben pentru defecte minore
            elif defect.severity < 0.7:
                color = (0, 165, 255)  # portocaliu pentru defecte moderate
            
            cv2.circle(image, defect.position, 10, color, 2)
            cv2.putText(image, defect.defect_type.value, 
                       (defect.position[0] + 15, defect.position[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        status_color = (0, 255, 0) if result.is_valid else (0, 0, 255)
        
        cv2.putText(image, result.quality_level, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, result.status_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        
        cv2.imwrite(output_path, image)
    
    def generate_report(self, result: QualityResult) -> str:
        """Generează un raport detaliat"""
        report = f"""
╔══════════════════════════════════════════════════════════╗
║        RAPORT CONTROL CALITATE ANVELOPE                  ║
╚══════════════════════════════════════════════════════════╝

    STATUS: {result.status_message}
    NIVEL CALITATE: {result.quality_level}
    DECIZIE: {'ACCEPTAT ✓' if result.is_valid else 'RESPINS ✗'}

    REZUMAT:
   {result.summary}

    Timp procesare: {result.processing_time:.3f}s
    Linii detectate: {len(result.detected_lines)}/{len(self.current_pattern.colors)}

"""
        
        if result.defects:
            report += f"  DEFECTE DETECTATE ({len(result.defects)}):\n"
            report += "─" * 60 + "\n"
            
            for i, defect in enumerate(result.defects, 1):
                severity_bar = "█" * int(defect.severity * 10) + "░" * (10 - int(defect.severity * 10))
                severity_text = "CRITIC" if defect.severity > 0.7 else "MODERAT" if defect.severity > 0.3 else "MINOR"
                
                report += f"""
{i}. {defect.defect_type.value.upper()} - {severity_text}
   Severitate: [{severity_bar}] {defect.severity:.2f}
   Poziție: ({defect.position[0]}, {defect.position[1]})
   Descriere: {defect.description}
   Încredere detecție: {defect.confidence:.0%}
"""
        else:
            report += " NICIUN DEFECT DETECTAT - Pattern perfect!\n"
        
        if result.detected_lines:
            report += "\n DETALII LINII DETECTATE:\n"
            report += "─" * 60 + "\n"
            for color, info in result.detected_lines.items():
                report += f"""
 {color.upper()}:
   • Poziție X: {info['x_position']}px
   • Lățime: {info['width']}px
   • Înălțime: {info['height']}px
   • Raport aspect: {info['aspect_ratio']:.2f}
   • Încredere detecție: {info['confidence']:.0%}
"""
        
        report += "\n" + "═" * 60 + "\n"
        
        return report