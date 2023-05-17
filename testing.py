import numpy as np
import matplotlib.pyplot as plt
from spectral import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QComboBox, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap


class HyperspectralTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hyperspectral Visualization Tool")

        self.image_path = ""
        self.selected_channel = None
        self.export_path = ""

        # Load button
        self.load_button = QPushButton("Load Hyperspectral Image", self)
        self.load_button.clicked.connect(self.load_image)

        # Channel selection
        self.channel_label = QLabel("Select Channel:")
        self.channel_combo = QComboBox(self)
        self.channel_combo.currentIndexChanged.connect(self.select_channel)

        # Display button
        self.display_button = QPushButton("Display Channel", self)
        self.display_button.clicked.connect(self.display_channel)

        # Convert to RGB button
        self.rgb_button = QPushButton("Convert to RGB", self)
        self.rgb_button.clicked.connect(self.convert_to_rgb)

        # Export button
        self.export_button = QPushButton("Export Channel", self)
        self.export_button.clicked.connect(self.export_channel)

        # ROI button
        self.roi_button = QPushButton("Select ROI", self)
        self.roi_button.clicked.connect(self.select_roi)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.channel_label)
        layout.addWidget(self.channel_combo)
        layout.addWidget(self.display_button)
        layout.addWidget(self.rgb_button)
        layout.addWidget(self.export_button)
        layout.addWidget(self.roi_button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def load_image(self):
        file_dialog = QFileDialog()
        self.image_path, _ = file_dialog.getOpenFileName(self, "Load Hyperspectral Image")
        if self.image_path:
            self.channel_combo.clear()
            self.selected_channel = None

            hyper_image = open_image(self.image_path)
            self.num_channels = hyper_image.shape[2]

            self.channel_combo.addItems([str(i) for i in range(self.num_channels)])

    def select_channel(self, index):
        self.selected_channel = index

    def display_channel(self):
        if self.image_path and self.selected_channel is not None:
            hyper_image = open_image(self.image_path)
            selected_band = hyper_image.read_band(self.selected_channel)

            plt.imshow(selected_band, cmap='gray')
            plt.title("Channel {}".format(self.selected_channel))
            plt.show()

    def convert_to_rgb(self):
        if self.image_path:
            hyper_image = open_image(self.image_path)
            rgb_image = hyper_image.rgb()

            plt.imshow(rgb_image)
            plt.title("RGB Image")
            plt.show()

    def export_channel(self):
        if self.image_path and self.selected_channel is not None:
            hyper_image = open_image(self.image_path)
            selected_band = hyper_image.read_band(self.selected_channel)

            file_dialog = QFileDialog()
            self.export_path, _ = file_dialog.getSaveFileName(self, "Export Channel")
            if self.export_path:
                export_image = selected_band.astype(np.uint8)
                plt.imsave(self.export_path, export_image, cmap='gray')
                print("Exported channel saved as", self.export_path)

    def select_roi(self):
        if self.image_path:
            hyper_image = open_image(self.image_path)
            rgb_image = hyper_image.rgb()

            plt.imshow(rgb_image)
            roi = plt.ginput(n=-1, timeout=-1)

            plt.close()

            # Extract spectral signatures from the ROI
            spectral_signatures = hyper_image.spectral_roi(roi)

            # Plot spectral signatures
            plt.plot(hyper_image.bands.centers, spectral_signatures.T)
            plt.xlabel('Wavelength')
            plt.ylabel('Intensity')
            plt.title('Spectral Signatures')
            plt.show()

if __name__ == '__main__':
    app = QApplication([])
    window = HyperspectralTool()
    window.show()
    app.exec_()
