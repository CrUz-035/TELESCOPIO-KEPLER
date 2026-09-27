import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QCompleter, QListWidget, QComboBox, QPushButton, QTextEdit, QSplitter,
    QAbstractItemView
)

from analisys import search_star, get_stardata

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

        self.star_input = QLineEdit()
        left_layout.addWidget(self.star_input)

        self.search_btn = QPushButton("Search")
        self.search_btn.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold; padding: 6px;")
        self.search_btn.clicked.connect(self.exceute_search)
        left_layout.addWidget(self.search_btn)

        label_list = QLabel("Records found:")
        label_list.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(label_list)

        self.results_list = QListWidget()
        self.results_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        left_layout.addWidget(self.results_list)



        self.logs_text = QTextEdit()
        self.logs_text.setPlaceholderText("")
        self.logs_text.setMaximumHeight(150)
        left_layout.addWidget(self.logs_text)

        #Placeholders

        left_layout.addWidget(QLabel("<b>Graphis Settings:</b>"))
        self.config_btn_placeholder = QPushButton("Transit Parameters")
        left_layout.addWidget(self.config_btn_placeholder)

        left_layout.addStretch()

        #Execute Analisys
        self.analyze_btn = QPushButton("Excecute Analisys")
        self.analyze_btn.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold; padding: 8px;")
        left_layout.addWidget(self.analyze_btn)
        #-----------------------------------------

        #-----------------------------------------
        #Right Side
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container) 

        #Graphic (Add function)
        label_graphic = QLabel("Light Curve Graphic")
        label_graphic.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(label_graphic)
        #Placeholders
        self.plot_placeholder = QLabel("Garphic")
        self.plot_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_placeholder.setStyleSheet("background-color: #1e1e1e; color: #aaaaaa; border: 1px solid #444; border-radius: 4px;")
        self.plot_placeholder.setMinimumHeight(350)
        right_layout.addWidget(self.plot_placeholder)

        #Analisys Results
        label_results = QLabel("Results:")
        label_results.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(label_results)
        self.results_text = QTextEdit()
        self.results_text.setPlaceholderText("Numeric and graphic analisis here")
        self.results_text.setMaximumHeight(150)
        right_layout.addWidget(self.results_text)

        #Export Analisys
        self.export_btn = QPushButton("Export")
        right_layout.addWidget(self.export_btn)
        #-----------------------------------------

        #Load Splitters and set proportion
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        
        # Proporción inicial de la división (Izquierda 35%, Derecha 65%)
        splitter.setSizes([420, 780])


        self.current_search_result = None


    def exceute_search(self):
        target = self.star_input.text().strip()
        if not target:
            self.logs_text.append("Invalid input.")
            return

        self.logs_text.append(f"Searching...")
        QApplication.processEvents()

        try:
            self.current_search_result = search_star(target)
            self.results_list.clear()
            if(self.current_search_result) == 0:
                self.logs_text.append(f"No results for '{target}'.")
                return

            for i, row in enumerate(self.current_search_result):
                item_text = f"[{i}] Mission: {row.mission} | Year: {row.year} | Author: {row.author} | Exptime: {row.exptime}s"
                self.results_list.addItem(item_text)

            self.logs_text.append(f"{len(self.current_search_result)} records found.")

        except Exception as e:
            self.logs_text.append("Search error.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

