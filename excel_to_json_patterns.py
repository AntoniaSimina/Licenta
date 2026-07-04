import pandas as pd
import json
import os
import re


def normalize_cell(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_pattern_name(value):
    raw = normalize_cell(value)
    if raw.startswith(':'):
        raw = raw[1:]
    return "".join(ch for ch in raw.upper() if ch.isalpha())


def extract_positions(raw_value):
    raw = normalize_cell(raw_value)
    if not raw:
        return []
    return re.findall(r"\d+", raw)


def extract_raw_colors(row):
    colors = []
    for col_idx in range(2, 6):
        color = normalize_cell(row[col_idx]).upper()
        if color and color != "NAN":
            colors.append(color)
    return colors


def pick_aligned_colors(raw_colors, official_pattern, positions):
    official_colors = list(official_pattern)
    target_len = len(positions)

    candidates = []
    if official_colors:
        candidates.append(("official", official_colors))
    if raw_colors:
        candidates.append(("raw", raw_colors))

    if not candidates:
        return []

    if target_len == 0:
        return candidates[0][1]

    source, best = min(candidates, key=lambda item: (abs(len(item[1]) - target_len), 0 if item[0] == "official" else 1))

    if len(best) > target_len:
        return best[:target_len]

    if len(official_colors) == target_len:
        return official_colors

    return best

def read_patterns_from_excel(excel_file):
    df = pd.read_excel(excel_file, header=None)
    
    patterns = []
    seen_recipe_ids = set()
    skipped_duplicates = 0
    mismatched_rows = 0
    
    for row_idx in range(4, len(df)):
        row = df.iloc[row_idx]
        
        recipe_id = normalize_cell(row[0])
        product_code = normalize_cell(row[1])
        
        if not recipe_id or recipe_id.upper() == "NAN":
            continue

        if recipe_id in seen_recipe_ids:
            skipped_duplicates += 1
            continue
        seen_recipe_ids.add(recipe_id)

        raw_colors = extract_raw_colors(row)

        positions = extract_positions(row[6])
        
        pattern_name_official = normalize_pattern_name(row[8])

        aligned_colors = pick_aligned_colors(raw_colors, pattern_name_official, positions)
        pattern_name_built = "".join(aligned_colors)

        if positions and len(aligned_colors) != len(positions):
            mismatched_rows += 1
            
        pattern_data = {
            "recipe_id": recipe_id,
            "product_code": product_code,
            "colors": aligned_colors,
            "pattern_name": pattern_name_built,
            "pattern_name_official": pattern_name_official,
            "positions_mm": positions
        }
        
        patterns.append(pattern_data)

    print(f"Info: duplicate recipe_id sarite: {skipped_duplicates}")
    print(f"Info: randuri cu mismatch culori/pozitii: {mismatched_rows}")
        
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
        print("Fisierul Excel nu a fost gasit!")