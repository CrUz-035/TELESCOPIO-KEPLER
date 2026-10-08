import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QCompleter, QListWidget, QComboBox, QPushButton, QTextEdit, QSplitter,
    QAbstractItemView, QCheckBox
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

        search_layout = QHBoxLayout()

        self.star_input = QLineEdit()
        search_layout.addWidget(self.star_input)

        # Mission Checkboxes
        self.chk_kepler = QCheckBox("Kepler/K2")
        self.chk_kepler.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_kepler.setChecked(True)
        search_layout.addWidget(self.chk_kepler)

        self.chk_tess = QCheckBox("TESS")
        self.chk_tess.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_tess.setChecked(True)
        search_layout.addWidget(self.chk_tess)

        self.search_btn = QPushButton("Search")
        self.search_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.search_btn.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold; padding: 6px;")
        self.search_btn.clicked.connect(self.exceute_search)
        search_layout.addWidget(self.search_btn)

        left_layout.addLayout(search_layout)


        #List of records founded
        label_list = QLabel("Records found:")
        label_list.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(label_list)

        self.results_list = QListWidget()
        self.results_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        left_layout.addWidget(self.results_list, stretch=1)

        self.results_list.itemSelectionChanged.connect(self.update_download_btn)


        #Logs box
        self.logs_text = QTextEdit()
        self.logs_text.setPlaceholderText("")
        self.logs_text.setMaximumHeight(150)
        line_height = self.logs_text.fontMetrics().lineSpacing()
        self.logs_text.setFixedHeight(line_height*3 + 12)
        self.logs_text.setReadOnly(True)
        left_layout.addWidget(self.logs_text)

        #Execute Analisys
        self.download_btn = QPushButton("Download data")
        self.download_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.download_btn.setEnabled(False)
        self.download_btn.clicked.connect(self.excecute_download)
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: #2b5b84; 
                color: white; 
                font-weight: bold; 
                padding: 8px;
            }
            QPushButton:disabled {
                background-color: #d3d3d3; 
                color: #7f7f7f;
            }
        """)
        left_layout.addWidget(self.download_btn)
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

        self.download_btn.setEnabled(False)

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

            self.update_download_btn()
        except Exception as e:
            self.logs_text.append(f"Search error: {str(e)}")
            self.download_btn.setEnabled(False)

    def excecute_download(self):
        return

    def update_download_btn(self):
        theres_results = len(self.results_list) > 0
        is_selected = len(self.results_list.selectedIndexes()) > 0

        print(theres_results and is_selected)

        self.download_btn.setEnabled(theres_results and is_selected)
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

