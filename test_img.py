import cv2
from advanced_tire_qc import AdvancedTireQualityChecker, DefectType

def main():
    checker = AdvancedTireQualityChecker()
    checker.set_current_pattern("YAWG")

    image_path = "test_shifted_small.jpg"   # ← poza ta
    image = cv2.imread(image_path)

    if image is None:
        print("Nu pot citi imaginea.")
        return

    # Analiză identică cu video
    defects, debug_info = checker._analyze_frame_absolute(image)

    print("\n=== REZULTAT ANALIZĂ IMAGINE ===")

    if not defects:
        print("✔ NU s-au detectat defecte.")
    else:
        for d in defects:
            print(f"✖ {d.defect_type.value}: {d.description}")

    # -----------------------------
    # OVERLAY
    # -----------------------------
    overlay = image.copy()
    checker._draw_debug_overlay(overlay, debug_info)

    # desenăm DOAR defecte reale
    for d in defects:
        if d.defect_type != DefectType.LINE_SHIFTED:
            continue

        cv2.circle(
            overlay,
            d.position,
            10,
            (0, 0, 255),
            2
        )

    cv2.imwrite("output_debug_image.png", overlay)
    print("\nImagine salvată ca output_debug_image.png")

    cv2.imshow("Debug QC", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
