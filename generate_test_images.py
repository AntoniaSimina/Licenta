# generate_test_images.py
"""
Generator simplu de imagini test pentru sistemul de control calitate
Rulează acest script pentru a genera automat toate imaginile de test
"""

import cv2
import numpy as np
import os

def create_base_tire_image(width=400, height=600):
    """Creează imaginea de bază a cauciucului"""
    # Fundal gri închis (cauciuc)
    image = np.full((height, width, 3), (45, 45, 45), dtype=np.uint8)
    return image

def add_line(image, x, width, color, height=None, broken=False):
    """Adaugă o linie colorată pe imagine"""
    if height is None:
        height = image.shape[0]
    
    if not broken:
        # Linie continuă
        image[0:height, x:x+width] = color
    else:
        # Linie întreruptă
        segment_height = 40
        gap_height = 20
        
        current_y = 0
        while current_y < height:
            segment_end = min(current_y + segment_height, height)
            image[current_y:segment_end, x:x+width] = color
            current_y = segment_end + gap_height

def add_contamination(image, num_spots=5):
    """Adaugă pete de contaminare"""
    height, width = image.shape[:2]
    
    for _ in range(num_spots):
        # Poziție aleatoare
        center_x = np.random.randint(20, width - 20)
        center_y = np.random.randint(20, height - 20)
        radius = np.random.randint(3, 8)
        
        # Culoare maro pentru pete
        color = (19, 69, 139)  # BGR format
        cv2.circle(image, (center_x, center_y), radius, color, -1)

def add_noise(image, intensity=0.05):
    """Adaugă zgomot pentru realism"""
    noise = np.random.normal(0, 25, image.shape).astype(np.int16)
    noisy_image = image.astype(np.int16) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    # Aplică zgomotul doar la anumite pixeli
    mask = np.random.random(image.shape[:2]) < intensity
    result = image.copy()
    result[mask] = noisy_image[mask]
    
    return result

def generate_correct_image():
    """Generează imaginea corectă"""
    print("🟢 Generez imagine corectă...")
    
    image = create_base_tire_image()
    height, width = image.shape[:2]
    
    # Calculează pozițiile
    line_widths = [40, 35, 40]  # roșu, verde, albastru
    spacing = 50
    total_width = sum(line_widths) + 2 * spacing
    start_x = (width - total_width) // 2
    
    # Culori BGR (OpenCV format)
    colors = [
        (0, 0, 255),    # Roșu
        (0, 255, 0),    # Verde  
        (255, 0, 0)     # Albastru
    ]
    
    current_x = start_x
    for i, (line_width, color) in enumerate(zip(line_widths, colors)):
        add_line(image, current_x, line_width, color)
        if i < len(line_widths) - 1:  # Nu adăuga spacing după ultima linie
            current_x += line_width + spacing
    
    # Adaugă puțin zgomot pentru realism
    image = add_noise(image, 0.02)
    
    cv2.imwrite("tire_correct.png", image)
    print("✅ Salvat: tire_correct.png")

def generate_broken_red_image():
    """Generează imagine cu linia roșie întreruptă"""
    print("🔴 Generez imagine cu linie roșie întreruptă...")
    
    image = create_base_tire_image()
    height, width = image.shape[:2]
    
    line_widths = [40, 35, 40]
    spacing = 50
    total_width = sum(line_widths) + 2 * spacing
    start_x = (width - total_width) // 2
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    
    current_x = start_x
    for i, (line_width, color) in enumerate(zip(line_widths, colors)):
        broken = (i == 0)  # Prima linie (roșa) va fi întreruptă
        add_line(image, current_x, line_width, color, broken=broken)
        if i < len(line_widths) - 1:
            current_x += line_width + spacing
    
    image = add_noise(image, 0.03)
    
    cv2.imwrite("tire_broken_red.png", image)
    print("✅ Salvat: tire_broken_red.png")

def generate_broken_green_image():
    """Generează imagine cu linia verde întreruptă"""
    print("🟢 Generez imagine cu linie verde întreruptă...")
    
    image = create_base_tire_image()
    height, width = image.shape[:2]
    
    line_widths = [40, 35, 40]
    spacing = 50
    total_width = sum(line_widths) + 2 * spacing
    start_x = (width - total_width) // 2
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    
    current_x = start_x
    for i, (line_width, color) in enumerate(zip(line_widths, colors)):
        broken = (i == 1)  # A doua linie (verde) va fi întreruptă
        add_line(image, current_x, line_width, color, broken=broken)
        if i < len(line_widths) - 1:
            current_x += line_width + spacing
    
    image = add_noise(image, 0.03)
    
    cv2.imwrite("tire_broken_green.png", image)
    print("✅ Salvat: tire_broken_green.png")

def generate_wrong_spacing_image():
    """Generează imagine cu spațiere greșită"""
    print("📏 Generez imagine cu spațiere greșită...")
    
    image = create_base_tire_image()
    height, width = image.shape[:2]
    
    line_widths = [40, 35, 40]
    # Spațieri diferite - prima mică, a doua mare
    spacings = [25, 80]  # În loc de 50, 50
    
    total_width = sum(line_widths) + sum(spacings)
    start_x = (width - total_width) // 2
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    
    current_x = start_x
    for i, (line_width, color) in enumerate(zip(line_widths, colors)):
        add_line(image, current_x, line_width, color)
        if i < len(spacings):
            current_x += line_width + spacings[i]
    
    image = add_noise(image, 0.03)
    
    cv2.imwrite("tire_wrong_spacing.png", image)
    print("✅ Salvat: tire_wrong_spacing.png")

def generate_wrong_width_image():
    """Generează imagine cu lățimi greșite"""
    print("📐 Generez imagine cu lățimi greșite...")
    
    image = create_base_tire_image()
    height, width = image.shape[:2]
    
    # Lățimi greșite: roșu prea lat, verde normal, albastru prea îngust
    line_widths = [70, 35, 15]  # În loc de 40, 35, 40
    spacing = 50
    total_width = sum(line_widths) + 2 * spacing
    start_x = (width - total_width) // 2
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    
    current_x = start_x
    for i, (line_width, color) in enumerate(zip(line_widths, colors)):
        add_line(image, current_x, line_width, color)
        if i < len(line_widths) - 1:
            current_x += line_width + spacing
    
    image = add_noise(image, 0.03)
    
    cv2.imwrite("tire_wrong_width.png", image)
    print("✅ Salvat: tire_wrong_width.png")

def generate_contamination_image():
    """Generează imagine cu contaminare"""
    print("🦠 Generez imagine cu contaminare...")
    
    image = create_base_tire_image()
    height, width = image.shape[:2]
    
    # Adaugă liniile normale
    line_widths = [40, 35, 40]
    spacing = 50
    total_width = sum(line_widths) + 2 * spacing
    start_x = (width - total_width) // 2
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    
    current_x = start_x
    for i, (line_width, color) in enumerate(zip(line_widths, colors)):
        add_line(image, current_x, line_width, color)
        if i < len(line_widths) - 1:
            current_x += line_width + spacing
    
    # Adaugă contaminare
    add_contamination(image, 8)
    
    image = add_noise(image, 0.05)
    
    cv2.imwrite("tire_contamination.png", image)
    print("✅ Salvat: tire_contamination.png")

def generate_missing_color_image():
    """Generează imagine cu culoare lipsă"""
    print("❌ Generez imagine cu culoare lipsă...")
    
    image = create_base_tire_image()
    height, width = image.shape[:2]
    
    # Doar 2 linii în loc de 3 (linia verde lipsește)
    line_widths = [40, 40]  # Doar roșu și albastru
    spacing = 50
    total_width = sum(line_widths) + spacing
    start_x = (width - total_width) // 2
    
    colors = [(0, 0, 255), (255, 0, 0)]  # Doar roșu și albastru
    
    current_x = start_x
    for i, (line_width, color) in enumerate(zip(line_widths, colors)):
        add_line(image, current_x, line_width, color)
        if i < len(line_widths) - 1:
            current_x += line_width + spacing
    
    image = add_noise(image, 0.03)
    
    cv2.imwrite("tire_missing_color.png", image)
    print("✅ Salvat: tire_missing_color.png")

def main():
    """Funcția principală - generează toate imaginile"""
    print("🏭 GENERATOR IMAGINI TEST ANVELOPE")
    print("=" * 50)
    
    # Verifică dacă OpenCV este instalat
    try:
        cv2.__version__
    except:
        print("❌ OpenCV nu este instalat!")
        print("Rulează: pip install opencv-python")
        return
    
    print("📁 Generez imaginile în folderul curent...")
    print()
    
    # Generează toate imaginile
    generate_correct_image()
    generate_broken_red_image()
    generate_broken_green_image()
    generate_wrong_spacing_image()
    generate_wrong_width_image()
    generate_contamination_image()
    generate_missing_color_image()
    
    print()
    print("✅ GATA! Au fost generate următoarele imagini:")
    
    images = [
        "tire_correct.png - Imagine perfectă (ar trebui să fie VALIDĂ)",
        "tire_broken_red.png - Linie roșie întreruptă",
        "tire_broken_green.png - Linie verde întreruptă", 
        "tire_wrong_spacing.png - Spațiere incorectă",
        "tire_wrong_width.png - Lățimi incorecte",
        "tire_contamination.png - Pete și murdărie",
        "tire_missing_color.png - Culoare lipsă"
    ]
    
    for img in images:
        print(f"  📸 {img}")
    
    print()
    print(" Acum poți rula testul:")
    print("   python test_simple.py")
    print()
    print(" Rezultatele așteptate:")
    print("   tire_correct.png -> ✅ VALIDĂ (scor > 90)")
    print("   Restul imaginilor -> ❌ INVALIDE (cu defecte specifice)")

if __name__ == "__main__":
    main()