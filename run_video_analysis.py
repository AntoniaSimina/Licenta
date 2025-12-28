from advanced_tire_qc import AdvancedTireQualityChecker

def main():
    checker = AdvancedTireQualityChecker()
    checker.set_current_pattern("YAWG")
    checker.measure_actual_positions("reference.jpg")
    checker.debug_mode = True
    checker.fixed_tire_center_x=991
    rezultat = checker.analyze_video(
        # video_path=r"C:\\Users\\Antonia\\Downloads\\V20251202_105058_001.avi",
        # video_path="video_linie_deviata.avi",
        # video_path="video_linie_verde_deviata.avi",
        # video_path="video_linie_alba_intrerupta.avi",
        # video_path="video_linie_alba_culoare_gresita.avi",
        # video_path="video_linie_galbena_margini_neregulate.avi",
        video_path="video_linie_galbena_mai_groasa.avi",
        output_video_path="output_overlay.avi",
        roi=(299, 779, 666, 1313),
        frame_skip=2
    )

    print("=== REZULTAT ANALIZĂ VIDEO ===")
    # print(rezultat["summary"])
    # print("Frame-uri totale:", rezultat["total_frames"])
    # print("Frame-uri analizate:", rezultat["analyzed_frames"])

if __name__ == "__main__":
    main()
