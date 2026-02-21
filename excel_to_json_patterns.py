import pandas as pd
import json
import os

def read_patterns_from_excel(excel_file):
    # Citim Excel-ul fără header pentru a controla exact rândurile
    df = pd.read_excel(excel_file, header=None)
    
    patterns = []
    
    # Datele reale încep de la rândul 4 (index 4)
    # Rândurile 0-3 sunt antete, categorii și tipuri de date (Product, String, etc.)
    for row_idx in range(4, len(df)):
        row = df.iloc[row_idx]
        
        # 1. Extragem ID-ul și Codul de Produs
        recipe_id = str(row[0]).strip() if pd.notna(row[0]) else None
        product_code = str(row[1]).strip() if pd.notna(row[1]) else None
        
        # Sărim peste rândurile goale sau invalide
        if not recipe_id or recipe_id == "nan":
            continue

        # 2. Colour Marking - Culorile sunt în coloanele 2, 3, 4 și 5
        raw_colors = []
        for col_idx in range(2, 6):
            color = str(row[col_idx]).strip() if pd.notna(row[col_idx]) else ""
            if color and color != "nan":
                raw_colors.append(color)
        
        # Construim numele pattern-ului din inițiale (ex: YAWG)
        pattern_name_built = "".join(raw_colors)
        
        # 3. EX_INF_CodeColor - Pozițiile sunt în coloana 6
        raw_positions = str(row[6]).strip() if pd.notna(row[6]) else ""
        positions = [p.strip() for p in raw_positions.split('.') if p.strip()]
        
        # 4. Ex_INF_ID_Recipe - Numele din Excel este în coloana 8
        pattern_name_excel = str(row[8]).strip() if pd.notna(row[8]) else ""
        if pattern_name_excel.startswith(':'):
            pattern_name_excel = pattern_name_excel[1:] # Eliminăm ":"
            
        # Construim obiectul final pentru acest pattern
        pattern_data = {
            "recipe_id": recipe_id,
            "product_code": product_code,
            "colors": raw_colors,
            "pattern_name": pattern_name_built,
            "pattern_name_official": pattern_name_excel,
            "positions_mm": positions
        }
        
        patterns.append(pattern_data)
        
    return patterns

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Succes! Am salvat {len(data)} pattern-uri în {filename}")

if __name__ == "__main__":
    FILE_NAME = "Copy of Tread Ext 3_4_25.11.2025.xlsx"
    if os.path.exists(FILE_NAME):
        data = read_patterns_from_excel(FILE_NAME)
        save_to_json(data, "patterns_productie.json")
    else:
        print("Fișierul Excel nu a fost găsit!")