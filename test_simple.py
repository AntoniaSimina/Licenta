# test_simple.py
"""
Fișier simplu pentru testarea sistemului de control calitate
Analizează automat toate imaginile PNG din folderul curent
"""

import os
import glob
from advanced_tire_qc import AdvancedTireQualityChecker

def test_toate_imaginile():
    """Testează toate imaginile PNG din folderul curent"""
    
    print("🔍 SISTEM CONTROL CALITATE ANVELOPE")
    print("=" * 50)
    
    # Găsește toate imaginile PNG
    imagini = glob.glob("*.png")
    
    if not imagini:
        print(" Nu am găsit imagini PNG în folderul curent!")
        print("Asigură-te că ai imagini cu extensia .png")
        return
    
    print(f" Am găsit {len(imagini)} imagini:")
    for img in imagini:
        print(f"   - {img}")
    
    # Creează checker-ul
    checker = AdvancedTireQualityChecker()
    checker.debug_mode = True
    
    print(f"\n Încep analiza...")
    
    rezultate = []
    
    for imagine in imagini:
        print(f"\n{'='*60}")
        print(f"ANALIZEZ: {imagine}")
        print(f"{'='*60}")
        
        try:
            # Analizează imaginea
            result = checker.analyze_tire(imagine)
            
            # Afișează rezultatul cu mesaje clare
            status_icon = "✅" if result.is_valid else "❌"
            print(f"\n{status_icon} DECIZIE: {'ACCEPTAT' if result.is_valid else 'RESPINS'}")
            print(f" Nivel calitate: {result.quality_level}")
            print(f" Status: {result.status_message}")
            print(f" Rezumat: {result.summary}")
            print(f"  Timp procesare: {result.processing_time:.3f} secunde")
            print(f" Defecte găsite: {len(result.defects)}")
            
            # Listează defectele
            if result.defects:
                print(f"\n  DEFECTE DETECTATE:")
                for i, defect in enumerate(result.defects, 1):
                    severity_text = "🔴 CRITIC" if defect.severity > 0.7 else "🟠 MODERAT" if defect.severity > 0.3 else "🟡 MINOR"
                    print(f"   {i}. {severity_text} - {defect.defect_type.value.upper()}")
                    print(f"      Severitate: {defect.severity:.2f}/1.0")
                    print(f"      Poziție: {defect.position}")
                    print(f"      Detalii: {defect.description}")
                    print()
            else:
                print(" Niciun defect detectat - pattern perfect!")
            
            # Salvează imaginea cu defectele marcate
            nume_rezultat = f"analizat_{imagine}"
            checker.save_debug_image(imagine, result, nume_rezultat)
            print(f" Imaginea analizată salvată: {nume_rezultat}")
            
            # Salvează raportul
            raport = checker.generate_report(result)
            nume_raport = f"raport_{imagine.replace('.png', '.txt')}"
            with open(nume_raport, 'w', encoding='utf-8') as f:
                f.write(raport)
            print(f" Raport salvat: {nume_raport}")
            
            # Adaugă la rezultate
            rezultate.append({
                'imagine': imagine,
                'valid': result.is_valid,
                'nivel': result.quality_level,
                'defecte': len(result.defects),
                'mesaj': result.status_message
            })
            
        except Exception as e:
            print(f" EROARE la procesarea {imagine}: {e}")
            import traceback
            traceback.print_exc()
            rezultate.append({
                'imagine': imagine,
                'valid': False,
                'nivel': 'EROARE',
                'defecte': 'EROARE',
                'mesaj': str(e)
            })
    
    # Afișează statisticile finale
    print_statistici_finale(rezultate)

def print_statistici_finale(rezultate):
    """Afișează statisticile finale"""
    
    print(f"\n{'='*60}")
    print(" STATISTICI FINALE")
    print(f"{'='*60}")
    
    total = len(rezultate)
    acceptate = sum(1 for r in rezultate if r['valid'])
    respinse = total - acceptate
    
    if total > 0:
        print(f"\n SUMAR GENERAL:")
        print(f"   Total imagini procesate: {total}")
        print(f"   ✅ Acceptate: {acceptate} ({acceptate/total*100:.1f}%)")
        print(f"   ❌ Respinse: {respinse} ({respinse/total*100:.1f}%)")
        
        # Distribuție pe nivele de calitate
        nivele = {}
        for r in rezultate:
            nivel = r.get('nivel', 'NECUNOSCUT')
            nivele[nivel] = nivele.get(nivel, 0) + 1
        
        if nivele:
            print(f"\n📊 DISTRIBUȚIE CALITATE:")
            for nivel, count in sorted(nivele.items()):
                print(f"   {nivel}: {count} imagini ({count/total*100:.1f}%)")
        
        print(f"\n DETALII PE IMAGINE:")
        for r in rezultate:
            status_icon = "✅" if r['valid'] else "❌"
            defecte_str = str(r['defecte']) if isinstance(r['defecte'], int) else r['defecte']
            print(f"  {status_icon} {r['imagine']:30} | {r['nivel']:20} | {defecte_str} defecte")
    
    print(f"\n🎯 RECOMANDĂRI:")
    if acceptate == total:
        print("   ✅ Toate imaginile sunt acceptate! Sistemul funcționează perfect.")
    elif acceptate > total * 0.8:
        print("   ✅ Majoritatea imaginilor sunt acceptate. Verificați cele respinse.")
    elif acceptate > total * 0.5:
        print("   ⚠️  Aproximativ jumătate din imagini sunt acceptate.")
        print("   💡 Posibile probleme: calibrare, iluminare sau setări pattern.")
    else:
        print("   ❌ Multe imagini respinse!")
        print("   💡 Verificați urgent: configurația sistemului, range-uri culori, calibrare cameră.")

def test_o_singura_imagine(nume_imagine):
    """Testează o singură imagine specificată"""
    
    if not os.path.exists(nume_imagine):
        print(f"❌ Imaginea {nume_imagine} nu există!")
        return
    
    print(f"🔍 Testez doar imaginea: {nume_imagine}\n")
    
    checker = AdvancedTireQualityChecker()
    result = checker.analyze_tire(nume_imagine)
    
    status_icon = "✅" if result.is_valid else "❌"
    print(f"{status_icon} DECIZIE: {'ACCEPTAT' if result.is_valid else 'RESPINS'}")
    print(f" Nivel: {result.quality_level}")
    print(f" Status: {result.status_message}")
    print(f" {result.summary}")
    print(f" Defecte: {len(result.defects)}")
    
    # Afișează raportul complet
    raport = checker.generate_report(result)
    print("\n" + raport)
    
    # Salvează rezultatele
    try:
        nume_rezultat = f"analizat_{nume_imagine}"
        checker.save_debug_image(nume_imagine, result, nume_rezultat)
        print(f" Imaginea analizată salvată: {nume_rezultat}")
        
        nume_raport = f"raport_{nume_imagine.replace('.png', '.txt')}"
        with open(nume_raport, 'w', encoding='utf-8') as f:
            f.write(raport)
        print(f" Raport salvat: {nume_raport}")
    except Exception as e:
        print(f"  Nu s-au putut salva rezultatele: {e}")

if __name__ == "__main__":
    print("Alegeți opțiunea:")
    print("1. Testează toate imaginile PNG din folder")
    print("2. Testează o imagine specifică")
    
    try:
        optiune = input("Introduceți 1 sau 2: ").strip()
        
        if optiune == "1":
            test_toate_imaginile()
        elif optiune == "2":
            nume = input("Introduceți numele imaginii (ex: tire_correct.png): ").strip()
            test_o_singura_imagine(nume)
        else:
            print("Rulez testul pentru toate imaginile...")
            test_toate_imaginile()
    except KeyboardInterrupt:
        print("\n\n Test întrerupt de utilizator.")
    except Exception as e:
        print(f"\n Eroare: {e}")
    
    print("\n Gata! Verificați fișierele generate.")