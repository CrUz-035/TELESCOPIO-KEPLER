import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QCompleter, QListWidget, QComboBox, QPushButton, QTextEdit, QSplitter,
    QAbstractItemView, QCheckBox, QTableWidget, QTableWidgetItem, QButtonGroup, QFileDialog
)

from analisys import search_star, get_stardata
from transit_method import analyze_transit

class MainWindow(QMainWindow):
    def __init__(self):
        
        super().__init__()
        #Title and window size
        self.setWindowTitle("Exoplanet Detector")
        self.resize(1200,600)

        #Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        #Layout for 2 column window
        main_Layout = QHBoxLayout(central_widget)

        #Splitter that allows resizing window
        splitter = QSplitter()
        main_Layout.addWidget(splitter)

        #-----------------------------------------
        #Left Side
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)

        #Star Selector
        label_star = QLabel("Star Selector (Kepler/TESS):")
        label_star.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(label_star)

        search_layout = QHBoxLayout()

        self.star_input = QLineEdit()
        search_layout.addWidget(self.star_input)

        self.search_btn = QPushButton("Search")
        self.search_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.search_btn.clicked.connect(self.exceute_analisys)
        self.search_btn.setStyleSheet("""
            QPushButton {
                background-color: #2b5b84; 
                color: white; 
                font-weight: bold; 
                padding: 6px;
            }
            QPushButton:disabled {
                background-color: #d3d3d3; 
                color: #7f7f7f;
            }
        """)
        search_layout.addWidget(self.search_btn)

        left_layout.addLayout(search_layout)

        label_tabla = QLabel("Analisys results:")
        label_tabla.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(label_tabla)

        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True) 
        
        left_layout.addWidget(self.results_display, stretch=1) 

        #Logs box
        self.logs_text = QTextEdit()
        self.logs_text.setPlaceholderText("")
        self.logs_text.setMaximumHeight(150)
        line_height = self.logs_text.fontMetrics().lineSpacing()
        self.logs_text.setFixedHeight(line_height*3 + 12)
        self.logs_text.setReadOnly(True)
        left_layout.addWidget(self.logs_text)
        #-----------------------------------------

        #-----------------------------------------
        #Right Side
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container) 

        #Graphics
        label_graphic = QLabel("Light Curve Graphic")
        label_graphic.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(label_graphic)

        graph_btns_layout = QHBoxLayout()
        
        self.graph0_btn = QPushButton("Raw Lightcurve")
        self.graph1_btn = QPushButton("Lightcurve")
        self.graph2_btn = QPushButton("Periodogram BLS")
        self.graph3_btn = QPushButton("Zoom")

        self.graph_btns = [
            self.graph0_btn, self.graph1_btn, self.graph2_btn, self.graph3_btn
        ]

        self.graph_btn_group = QButtonGroup(self)
        self.graph_btn_group.setExclusive(True)

        for i, btn in enumerate(self.graph_btns, start=1):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #e0e0e0; 
                    color: #333333; 
                    font-weight: bold; 
                    padding: 5px;
                    border-radius: 4px;
                }
                QPushButton:checked {
                    background-color: #2b5b84; 
                    color: white;
                }
            """)

            self.graph_btn_group.addButton(btn, i)
            graph_btns_layout.addWidget(btn)

        right_layout.addLayout(graph_btns_layout)

        self.graph_btn_group.idClicked.connect(self.select_graphics)

        self.graph0_btn.setChecked(True)

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas, stretch=1)

        #Export Analisys
        self.export_btn = QPushButton("Export")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #d8831c; 
                color: white; 
                font-weight: bold; 
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #d3d3d3; 
                color: #7f7f7f;
            }
        """)
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        right_layout.addWidget(self.export_btn)
        #-----------------------------------------

        #Load Splitters and set proportion
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        
        splitter.setSizes([420, 780])


        self.current_df = None
        self.current_plot_dict = None


    def exceute_analisys(self):
        target = self.star_input.text().strip()
        if not target:
            self.logs_text.clear()
            self.logs_text.append("Invalid input.")
            return

        self.search_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.logs_text.clear()

        def get_callback(message):
            self.logs_text.append(message)
            QApplication.processEvents()

        QApplication.processEvents()

        try:
            results, plot_data = analyze_transit(target, callback=get_callback) 

            self.current_df = results
            self.current_plot_dict = plot_data

            self.load_analisys_results(results)
            self.select_graphics(1)
            self.search_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
        except Exception as e:
            self.logs_text.append(f"Search error: {str(e)}")
            self.search_btn.setEnabled(True)

    def load_analisys_results(self, df: pd.DataFrame):
        self.results_display.clear()

        if df is None or df.empty:
            self.results_display.setPlainText("No data")
            return

        row = df.iloc[0]
        text_lines = []

        for col_name in df.columns:
            value = row[col_name]
            
            if isinstance(value, float):
                value_str = f"{value:.5f}"
            else:
                value_str = str(value)

            text_lines.append(f"• {col_name}: {value_str}")

        text_final = "\n".join(text_lines)
        self.results_display.setPlainText(text_final)

    def select_graphics(self, number):
            for i, btn in enumerate(self.graph_btns, start=1):
                btn.setChecked(i == number)
    
            if not self.current_plot_dict:
                return
    
            self.figure.clear()
            ax = self.figure.add_subplot(111)
    
            if number == 1:
                self.show_raw_lightcurve(ax, self.current_plot_dict)
            elif number == 2:
                self.show_lightcurve(ax, self.current_plot_dict)
            elif number == 3:
                self.show_periodograma_BLS(ax, self.current_plot_dict)
            elif number == 4:
                self.show_zoom(ax, self.current_plot_dict)
    
            self.canvas.draw()

    def show_raw_lightcurve(self, ax, data):
        t = data["time_raw"]
        f = data["flux_raw"]

        ax.plot(t, f, 'k.', markersize=2, alpha=0.6)
        ax.set_title(data["figures"]["raw"]["title"], fontsize=10, fontweight='bold')
        ax.set_xlabel(data["figures"]["raw"]["xlabel"], fontsize=9)
        ax.set_ylabel(data["figures"]["raw"]["ylabel"], fontsize=9)
        ax.legend(data["figures"]["raw"]["legend"])
        ax.grid(True, linestyle='--', alpha=0.5)

        self.canvas.draw()

    def show_lightcurve(self, ax, data):
        t_clean = data.get("time_clean", [])
        f_clean = data.get("flux_clean", [])
        t_binned = data.get("time_binned", [])
        f_binned = data.get("flux_binned", [])
        transit_times = data.get("transit_times", [])

        ax.plot(t_clean, f_clean, '.', color='gray', markersize=2, alpha=0.5, label='Clean')

        if len(t_binned) > 0 and len(f_binned) > 0:
            ax.plot(t_binned, f_binned, '.', color='blue', markersize=4, label='Binned')


        ax.set_title(data["figures"]["clean"]["title"], fontsize=10, fontweight='bold')
        ax.set_xlabel(data["figures"]["clean"]["xlabel"], fontsize=9)
        ax.set_ylabel(data["figures"]["clean"]["ylabel"], fontsize=9)
        ax.legend(data["figures"]["clean"]["legend"])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8)

        self.canvas.draw()

    def show_periodograma_BLS(self, ax, data):
        periods = data.get("periods", [])
        power = data.get("power", [])
        best_period = data.get("best_period", None)
        best_power = data.get("best_power", None)

        ax.plot(periods, power, color='black', linewidth=1)

        if best_period is not None and best_power is not None:
            ax.axvline(x=best_period, color='red', linestyle=':', label=f"Best Period: {best_period:.4f} d")
            ax.plot(best_period, best_power, 'ro', markersize=6)

        ax.set_title(data["figures"]["periodogram"]["title"], fontsize=10, fontweight='bold')
        ax.set_xlabel(data["figures"]["periodogram"]["xlabel"], fontsize=9)
        ax.set_ylabel(data["figures"]["periodogram"]["ylabel"], fontsize=9)
        ax.legend(data["figures"]["periodogram"]["legend"])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8)

    def show_zoom(self, ax, data):
        phase_h = data.get("phase_hours", [])
        folded_f = data.get("folded_flux", [])
        binned_phase_h = data.get("binned_phase_hours", [])
        binned_folded_f = data.get("binned_folded_flux", [])
        binned_folded_err = data.get("binned_folded_err", None)
        duration_h = data.get("duration_h", 0)
        window_h = data.get("window_h", 0)
        depth_ppm = data.get("depth_ppm", 0)

        ax.plot(phase_h, folded_f, '.', color='gray', markersize=2, alpha=0.4)

        if len(binned_phase_h) > 0 and len(binned_folded_f) > 0:
            if binned_folded_err is not None:
                ax.errorbar(binned_phase_h, binned_folded_f, yerr=binned_folded_err, 
                            fmt='ro', markersize=4, capsize=2, label='Binned mean')
            else:
                ax.plot(binned_phase_h, binned_folded_f, 'ro', markersize=4, label='Binned mean')

        if duration_h > 0:
            ax.axvspan(-duration_h / 2, duration_h / 2, color='orange', alpha=0.2, label='Transit duration')

        if window_h > 0:
            ax.set_xlim(-window_h, window_h)

        ax.set_title(data["figures"]["folded"]["title"], fontsize=10, fontweight='bold')
        ax.set_xlabel(data["figures"]["folded"]["xlabel"], fontsize=9)
        ax.set_ylabel(data["figures"]["folded"]["ylabel"], fontsize=9)
        ax.legend(data["figures"]["folded"]["legend"])
        ax.grid(True, linestyle='--', alpha=0.5)


    def export_results(self):
        self.export_btn.setEnabled(False)
        if self.current_df is None or self.current_plot_dict is None:
            self.logs_text.append("No data to export")
            return

        base_root = QFileDialog.getExistingDirectory(self, "Select path")
        if not base_root:
            return 

        try:
            target_name = self.star_input.text().strip().replace(" ", "_")
            folder_name = f"Results_{target_name}"
            folder_path = os.path.join(base_root, folder_name)

            os.makedirs(folder_path, exist_ok=True)

            txt_path = os.path.join(folder_path, "results.txt")
            row = self.current_df.iloc[0]
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"=== ANALYSIS RESULTS: {self.star_input.text().strip()} ===\n\n")
                for col_name in self.current_df.columns:
                    val = row[col_name]
                    f.write(f"• {col_name}: {val}\n")

            from matplotlib.figure import Figure

            graph_names = [
                ("01_raw_lightcurve.png", self.show_raw_lightcurve),
                ("02_lightcurve.png", self.show_lightcurve),
                ("03_periodogram_BLS.png", self.show_periodograma_BLS),
                ("04_zoom.png", self.show_zoom)
            ]

            for file_name, plt_func in graph_names:
                fig_temp = Figure(figsize=(8, 6), dpi=150)
                ax_temp = fig_temp.add_subplot(111)

                plt_func(ax_temp, self.current_plot_dict)

                path_img = os.path.join(folder_path, file_name)
                fig_temp.savefig(path_img, bbox_inches='tight')

            self.logs_text.append(f"Exported correctly:\n   {folder_path}\n")
            
            self.export_btn.setEnabled(True)

        except Exception as e:
            self.logs_text.append(f"Failed to export: {str(e)}\n")
            self.export_btn.setEnabled(True)

    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

