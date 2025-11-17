# simple_gui.py
"""
Interfață grafică simplă pentru testarea sistemului de control calitate
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk

try:
    from advanced_tire_qc import AdvancedTireQualityChecker
except ImportError:
    messagebox.showerror("Eroare", "Nu s-a putut importa advanced_tire_qc.py!\nAsigură-te că fișierul există în același folder.")
    exit()

class SimpleTireQualityGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Control Calitate Anvelope - Interfață Simplă")
        self.root.geometry("1000x700")
        
        self.checker = AdvancedTireQualityChecker()
        self.current_image_path = None
        self.result = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """Configurează interfața grafică"""
        
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame pentru butoane
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(buttons_frame, text="📁 Încarcă Imagine", 
                  command=self.load_image, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="🔍 Analizează", 
                  command=self.analyze_image, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="💾 Salvează Rezultat", 
                  command=self.save_result, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="📊 Analizează Folder", 
                  command=self.analyze_folder, width=20).pack(side=tk.LEFT, padx=5)
        
        # Frame pentru conținut
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame stânga - Imagine
        left_frame = ttk.LabelFrame(content_frame, text="Imagine")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Canvas pentru imagine
        self.image_canvas = tk.Canvas(left_frame, bg='lightgray')
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label pentru numele imaginii
        self.image_name_label = ttk.Label(left_frame, text="Nicio imagine încărcată", 
                                         font=('Arial', 10, 'bold'))
        self.image_name_label.pack(pady=5)
        
        # Frame dreapta - Rezultate
        right_frame = ttk.LabelFrame(content_frame, text="Rezultate Analiză")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Status frame
        status_frame = ttk.Frame(right_frame)
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(status_frame, text="Status:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, text="Neanalizat", 
                                     font=('Arial', 12, 'bold'), foreground='gray')
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Score frame
        score_frame = ttk.Frame(right_frame)
        score_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(score_frame, text="Scor Calitate:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.score_label = ttk.Label(score_frame, text="--/100", font=('Arial', 12, 'bold'))
        self.score_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Progress bar pentru scor
        self.score_progress = ttk.Progressbar(score_frame, length=200, mode='determinate')
        self.score_progress.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Defects frame
        defects_frame = ttk.LabelFrame(right_frame, text="Defecte Detectate")
        defects_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview pentru defecte
        columns = ('Tip', 'Severitate', 'Poziție', 'Descriere')
        self.defects_tree = ttk.Treeview(defects_frame, columns=columns, show='headings', height=8)
        
        # Configurează coloanele
        self.defects_tree.heading('Tip', text='Tip Defect')
        self.defects_tree.heading('Severitate', text='Severitate')
        self.defects_tree.heading('Poziție', text='Poziție')
        self.defects_tree.heading('Descriere', text='Descriere')
        
        self.defects_tree.column('Tip', width=120)
        self.defects_tree.column('Severitate', width=80)
        self.defects_tree.column('Poziție', width=80)
        self.defects_tree.column('Descriere', width=200)
        
        # Scrollbar pentru treeview
        scrollbar = ttk.Scrollbar(defects_frame, orient=tk.VERTICAL, command=self.defects_tree.yview)
        self.defects_tree.configure(yscrollcommand=scrollbar.set)
        
        self.defects_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
        # Frame pentru raport detaliat
        report_frame = ttk.LabelFrame(right_frame, text="Raport Detaliat")
        report_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Text widget pentru raport
        self.report_text = tk.Text(report_frame, height=6, wrap=tk.WORD, font=('Consolas', 9))
        report_scrollbar = ttk.Scrollbar(report_frame, orient=tk.VERTICAL, command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=report_scrollbar.set)
        
        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        report_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
    
    def load_image(self):
        """Încarcă o imagine pentru analiză"""
        file_path = filedialog.askopenfilename(
            title="Selectați imaginea pentru analiză",
            filetypes=[
                ("Imagini", "*.png *.jpg *.jpeg *.bmp"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("Toate fișierele", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.image_name_label.config(text=os.path.basename(file_path))
            self.clear_results()
    
    def display_image(self, image_path):
        """Afișează imaginea pe canvas"""
        try:
            # Deschide și redimensionează imaginea
            pil_image = Image.open(image_path)
            
            # Obține dimensiunile canvas-ului
            self.image_canvas.update()
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            # Redimensionează imaginea păstrând proporțiile
            if canvas_width > 1 and canvas_height > 1:
                pil_image.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
            else:
                pil_image.thumbnail((400, 300), Image.Resampling.LANCZOS)
            
            # Convertește pentru tkinter
            self.photo_image = ImageTk.PhotoImage(pil_image)
            
            # Afișează pe canvas
            self.image_canvas.delete("all")
            self.image_canvas.create_image(
                canvas_width//2 if canvas_width > 1 else 200,
                canvas_height//2 if canvas_height > 1 else 150,
                image=self.photo_image,
                anchor=tk.CENTER
            )
            
        except Exception as e:
            messagebox.showerror("Eroare", f"Nu s-a putut încărca imaginea:\n{e}")
    
    def analyze_image(self):
        """Analizează imaginea curentă"""
        if not self.current_image_path:
            messagebox.showwarning("Atenție", "Încărcați mai întâi o imagine!")
            return
        
        try:
            # Schimbă cursorul pentru a indica procesarea
            self.root.config(cursor="wait")
            self.root.update()
            
            # Analizează imaginea
            self.result = self.checker.analyze_tire(self.current_image_path)
            
            # Afișează rezultatele
            self.display_results()
            
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la analiza imaginii:\n{e}")
        finally:
            # Restaurează cursorul
            self.root.config(cursor="")
    
    def display_results(self):
        """Afișează rezultatele analizei"""
        if not self.result:
            return
        
        # Status
        if self.result.is_valid:
            self.status_label.config(text=f"✅ {self.result.quality_level}", foreground='green')
        else:
            self.status_label.config(text=f"❌ {self.result.quality_level}", foreground='red')
        
        # Mesaj status
        self.score_label.config(text=self.result.status_message)
        
        # Progress bar bazat pe nivel calitate
        quality_scores = {
            "EXCELENT": 100,
            "FOARTE BUN": 90,
            "BUN": 75,
            "ACCEPTABIL CU REZERVE": 60,
            "INACCEPTABIL": 30,
            "EROARE": 0
        }
        self.score_progress['value'] = quality_scores.get(self.result.quality_level, 50)
        
        # Curăță defectele anterioare
        for item in self.defects_tree.get_children():
            self.defects_tree.delete(item)
        
        # Adaugă defectele
        for defect in self.result.defects:
            severity_str = f"{defect.severity:.2f}"
            position_str = f"({defect.position[0]}, {defect.position[1]})"
            
            self.defects_tree.insert('', tk.END, values=(
                defect.defect_type.value,
                severity_str,
                position_str,
                defect.description[:50] + "..." if len(defect.description) > 50 else defect.description
            ))
        
        # Raport detaliat
        report = self.checker.generate_report(self.result)
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, report)
    
    def clear_results(self):
        """Curăță rezultatele anterioare"""
        self.status_label.config(text="Neanalizat", foreground='gray')
        self.score_label.config(text="--/100")
        self.score_progress['value'] = 0
        
        for item in self.defects_tree.get_children():
            self.defects_tree.delete(item)
        
        self.report_text.delete(1.0, tk.END)
    
    def save_result(self):
        """Salvează rezultatul analizei"""
        if not self.result or not self.current_image_path:
            messagebox.showwarning("Atenție", "Nu există rezultate de salvat!")
            return
        
        try:
            # Salvează imaginea cu defectele marcate
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            
            # Imaginea analizată
            analyzed_image_path = f"analizat_{base_name}.png"
            self.checker.save_debug_image(self.current_image_path, self.result, analyzed_image_path)
            
            # Raportul
            report_path = f"raport_{base_name}.txt"
            report = self.checker.generate_report(self.result)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            messagebox.showinfo("Succes", 
                f"Rezultatele au fost salvate:\n• {analyzed_image_path}\n• {report_path}")
            
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la salvarea rezultatelor:\n{e}")
    
    def analyze_folder(self):
        """Analizează toate imaginile dintr-un folder"""
        folder_path = filedialog.askdirectory(title="Selectați folderul cu imagini")
        
        if not folder_path:
            return
        
        # Găsește imaginile
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        image_files = []
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, file))
        
        if not image_files:
            messagebox.showwarning("Atenție", "Nu s-au găsit imagini în folderul selectat!")
            return
        
        # Fereastră de progres
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Analizez folder...")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        ttk.Label(progress_window, text="Analizez imaginile...").pack(pady=10)
        
        progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
        progress_bar.pack(pady=10)
        progress_bar['maximum'] = len(image_files)
        
        progress_label = ttk.Label(progress_window, text="")
        progress_label.pack(pady=5)
        
        # Analizează fiecare imagine
        results = []
        
        for i, image_file in enumerate(image_files):
            try:
                progress_label.config(text=f"Analizez: {os.path.basename(image_file)}")
                progress_window.update()
                
                result = self.checker.analyze_tire(image_file)
                results.append({
                    'file': os.path.basename(image_file),
                    'path': image_file,
                    'valid': result.is_valid,
                    'score': result.score,
                    'defects': len(result.defects),
                    'result': result
                })
                
                # Salvează rezultatele
                base_name = os.path.splitext(os.path.basename(image_file))[0]
                analyzed_path = os.path.join(folder_path, f"analizat_{base_name}.png")
                report_path = os.path.join(folder_path, f"raport_{base_name}.txt")
                
                self.checker.save_debug_image(image_file, result, analyzed_path)
                
                report = self.checker.generate_report(result)
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                
            except Exception as e:
                results.append({
                    'file': os.path.basename(image_file),
                    'path': image_file,
                    'valid': False,
                    'score': 0,
                    'defects': 'EROARE',
                    'error': str(e)
                })
            
            progress_bar['value'] = i + 1
            progress_window.update()
        
        progress_window.destroy()
        
        # Afișează raportul final
        self.show_batch_results(results, folder_path)
    
    def show_batch_results(self, results, folder_path):
        """Afișează rezultatele analizei batch"""
        
        # Fereastră pentru rezultate
        results_window = tk.Toplevel(self.root)
        results_window.title("Rezultate Analiză Folder")
        results_window.geometry("800x600")
        
        # Frame pentru statistici
        stats_frame = ttk.LabelFrame(results_window, text="Statistici")
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        total = len(results)
        valid = sum(1 for r in results if r['valid'])
        invalid = total - valid
        avg_score = sum(r['score'] for r in results if isinstance(r['score'], (int, float))) / total if total > 0 else 0
        
        stats_text = f"Total imagini: {total} | Valide: {valid} | Invalide: {invalid} | Scor mediu: {avg_score:.1f}/100"
        ttk.Label(stats_frame, text=stats_text, font=('Arial', 10, 'bold')).pack(pady=10)
        
        # Tabelul cu rezultate
        table_frame = ttk.LabelFrame(results_window, text="Rezultate Detaliate")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        columns = ('Imagine', 'Status', 'Scor', 'Defecte')
        results_tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        for col in columns:
            results_tree.heading(col, text=col)
        
        results_tree.column('Imagine', width=200)
        results_tree.column('Status', width=100)
        results_tree.column('Scor', width=100)
        results_tree.column('Defecte', width=100)
        
        # Adaugă rezultatele
        for result in results:
            status = "✅ VALIDĂ" if result['valid'] else "❌ INVALIDĂ"
            score = f"{result['score']:.1f}/100" if isinstance(result['score'], (int, float)) else "EROARE"
            defects = str(result['defects'])
            
            results_tree.insert('', tk.END, values=(result['file'], status, score, defects))
        
        # Scrollbar pentru tabel
        table_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=results_tree.yview)
        results_tree.configure(yscrollcommand=table_scrollbar.set)
        
        results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        table_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
        # Buton pentru închidere
        ttk.Button(results_window, text="Închide", command=results_window.destroy).pack(pady=10)
        
        # Salvează raportul sumar
        summary_path = os.path.join(folder_path, "raport_sumar_folder.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"RAPORT SUMAR ANALIZĂ FOLDER\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Folder analizat: {folder_path}\n")
            f.write(f"Total imagini: {total}\n")
            f.write(f"Imagini valide: {valid} ({valid/total*100:.1f}%)\n")
            f.write(f"Imagini invalide: {invalid} ({invalid/total*100:.1f}%)\n")
            f.write(f"Scor mediu: {avg_score:.1f}/100\n\n")
            
            f.write("DETALII PE IMAGINE:\n")
            f.write("-" * 50 + "\n")
            for result in results:
                status_text = "VALIDĂ" if result['valid'] else "INVALIDĂ"
                f.write(f"{result['file']}: {status_text} - {result['score']:.1f}/100 - {result['defects']} defecte\n")
        
        messagebox.showinfo("Succes", f"Analiza completă!\nRaport sumar salvat: {summary_path}")
    
    def run(self):
        """Pornește interfața grafică"""
        self.root.mainloop()

if __name__ == "__main__":
    app = SimpleTireQualityGUI()
    app.run()