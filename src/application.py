import sys
from PyQt6.QtWidgets import *
from PyQt6.QtWidgets import QWidget, QMessageBox, QApplication
from PyQt6.QtCore import QSize, QCoreApplication, Qt
from PyQt6.QtGui import QPixmap, QPalette
import qt_material


class SelectModelWindow(QWidget):
    def __init__(self, map):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel("Choose your model")

        dummy_btn = QPushButton("Dummy", self)
        dummy_btn.setFixedSize(100, 60)

        intermediate_btn = QPushButton("Intermediate", self)
        intermediate_btn.setFixedSize(100, 60)

        expert_btn = QPushButton("Expert", self)
        expert_btn.setFixedSize(100, 60)
        expert_btn.setIconSize(QSize(40, 40))

        layout.addWidget(dummy_btn)
        layout.addWidget(intermediate_btn)
        layout.addWidget(expert_btn)

        self.map = map

        layout.addWidget(self.label)
        self.setLayout(layout)


class ApplicationMangager(QWidget):
    def __init__(self):
        super().__init__()

        self.initialize()

    def initialize(self):
        self.select_model_window = None
        self.setGeometry(800, 200, 400, 300)
        self.setStyleSheet('QLabel{font-size:20pt; font-family:"Georgia"}')

        defend_the_center_btn = QPushButton("Defend the center", self)
        defend_the_center_btn.setFixedSize(200, 60)

        corridor_btn = QPushButton("Corridor", self)
        corridor_btn.setFixedSize(100, 60)

        deathmatch_btn = QPushButton("Deathmatch", self)
        deathmatch_btn.setFixedSize(100, 60)
        deathmatch_btn.setIconSize(QSize(40, 40))

        layout = QVBoxLayout()
        self.label = QLabel("Choose your map", parent=self)
        self.pixmap = QPixmap("Doom.jpg")
        self.pallete = QPalette()
        self.pallete.setBrush(QPalette.window(), self.pixmap)
        self.setPalette(self.palatte)

        layout.addWidget(self.label)
        layout.addWidget(defend_the_center_btn)
        layout.addWidget(corridor_btn)
        layout.addWidget(deathmatch_btn)

        layout.setAlignment(self.label, Qt.AlignmentFlag.AlignHCenter)
        layout.setAlignment(defend_the_center_btn, Qt.AlignmentFlag.AlignHCenter)
        layout.setAlignment(corridor_btn, Qt.AlignmentFlag.AlignHCenter)
        layout.setAlignment(deathmatch_btn, Qt.AlignmentFlag.AlignHCenter)

        defend_the_center_btn.clicked.connect(
            lambda: self.run_model("defend_the_center")
        )
        corridor_btn.clicked.connect(lambda: self.run_model("corridor"))
        deathmatch_btn.clicked.connect(lambda: self.run_model("deathmatch"))

        # self.setGeometry(300, 300, 350, 250)
        self.setLayout(layout)
        self.setWindowTitle("Doom Agent")
        self.show()

    def run_model(self, map):
        self.select_model_window = SelectModelWindow(map)
        self.select_model_window.show()
        print(f"Invoke Defend the center {map}")


def main():
    app = QApplication(sys.argv)
    # qt_material.apply_stylesheet(app, theme="dark_purple.xml")
    ex = ApplicationMangager()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
