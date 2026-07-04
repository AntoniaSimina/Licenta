# Smart Tire Quality Control

## Descriere
Smart Tire Quality Control este o aplicație desktop pentru analiza video și controlul calității anvelopelor, bazată pe procesare de imagini. Proiectul folosește un GUI în Python pentru încărcarea unui video local sau a unui stream RTSP, aplică calibrare cameră și omografie, apoi verifică poziția, lățimea și continuitatea liniilor/colorilor dintr-un pattern de producție.

## Tehnologii
- Python
- OpenCV
- Tkinter
- NumPy
- Pillow

## Instalare

```bash
pip install opencv-python numpy pillow
```

## Rulare

```bash
python app.py
```

## Structura proiectului

- app.py
- advanced_tire_qc.py
- calibrate_center_click.py
- calibrate_positions.py
- calibrate_widths.py
- excel_to_json_patterns.py
- find_colors_HSV.py
- matrice_omografie.npy
- modificam_poza.py
- patterns_productie.json
- roi_preview.py
- run_video_analysis_manual_tuning.py
- run_video_analysis.py
- simulam_poza_deviata.py

## Autor
Antonia AVRAM