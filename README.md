# Smart Tire Quality Control

## Descriere
Smart Tire Quality Control este o aplicație desktop pentru analiza video și controlul calității anvelopelor, bazată pe procesare de imagini. Aplicația încarcă un video local sau un stream RTSP, aplică calibrarea camerei și omografia, apoi verifică poziția, lățimea și continuitatea liniilor/colorilor dintr-un pattern de producție.

## Livrabilele proiectului
- Adresa repository-ului: https://github.com/AntoniaSimina/Licenta.git
- Codul sursă complet al aplicației este inclus în acest folder.
- Fișierele de configurare și datele folosite de aplicație sunt incluse în proiect.

## Tehnologii
- Python
- OpenCV
- Tkinter
- NumPy
- Pillow

## Cerințe
- Python 3.10 sau mai nou
- bibliotecile Python necesare instalate local

## Instalare

```bash
pip install opencv-python numpy pillow
```

## Lansarea aplicației

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