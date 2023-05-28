import numpy as np
import matplotlib.pyplot as plt
from spectral import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QComboBox, QPushButton, QFileDialog, QHBoxLayout, QMenu, QAction
from PyQt5.QtGui import QPixmap
import tifffile
import os
import imageio
from scipy.io import loadmat
spectral.settings.envi_support_nonlowercase_params = True
import matplotlib
matplotlib.use('Qt5Agg')

class HyperspectralTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hyperspectral Visualization Tool")

        self.image_path = ""
        self.selected_channel = None
        self.export_path = ""
        self.data = None

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

        # RGB button
        self.rgb_button = QPushButton("Convert to RGB", self)
        self.rgb_button.setMenu(self.create_rgb_menu())

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

    def select_channel(self, index):
        self.selected_channel = index

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Supported Files (*.hdr *.tiff *.mat)")
        if file_path:
            self.image_path = file_path
            self.channel_combo.clear()

            # Load hyperspectral image
            if file_path.endswith('.hdr'):
                self.data = spectral.open_image(file_path).load()  # Using imageio library for .hdr files
            elif file_path.endswith('.tiff'):
                self.data = plt.imread(file_path)  # Using matplotlib for .tiff files
            elif file_path.endswith('.mat'):
                mat_data = loadmat(file_path)  # Using scipy.io for .mat files
                print(mat_data.keys())  # Print the keys to identify the variable name
                # Use 'reflectances' as the variable name to access hyperspectral data
                self.data = mat_data['reflectances']
            else:
                # Handle unsupported file types or display an error message
                print("Unsupported file type.")

            # Get the number of channels
            num_channels = self.data.shape[-1]

            # Update channel combo box
            self.channel_combo.addItems([str(i) for i in range(1, num_channels + 1)])

            # Set default channel as the first channel
            self.channel_combo.setCurrentIndex(0)
            print("Image loaded successfully")

    def display_channel(self):
        if self.data is None or self.selected_channel is None:
            return

        channel_index = self.selected_channel
        selected_band = self.data[:, :, channel_index]

        # Normalize the pixel values to the range [0, 1]
        normalized_band = selected_band.astype(float) / np.max(selected_band)

        plt.imshow(normalized_band, cmap='gray')
        plt.axis('off')
        plt.title(f"Channel {channel_index + 1}")
        plt.show()

    def create_rgb_menu(self):
        menu = QMenu(self)
        red_action = QAction("Red Channel", self)
        green_action = QAction("Green Channel", self)
        blue_action = QAction("Blue Channel", self)

        red_action.triggered.connect(lambda: self.convert_to_rgb('r'))
        green_action.triggered.connect(lambda: self.convert_to_rgb('g'))
        blue_action.triggered.connect(lambda: self.convert_to_rgb('b'))

        menu.addAction(red_action)
        menu.addAction(green_action)
        menu.addAction(blue_action)

        return menu

    def convert_to_rgb(self, component):
        if self.data is None:
            return

        if component == 'r':
            rgb_data = self.data.read_band(25)
            rgb_data = np.stack((rgb_data, np.zeros_like(rgb_data), np.zeros_like(rgb_data)), axis=2)
        elif component == 'g':
            rgb_data = self.data.read_band(50)
            rgb_data = np.stack((np.zeros_like(rgb_data), rgb_data, np.zeros_like(rgb_data)), axis=2)
        elif component == 'b':
            rgb_data = self.data.read_band(75)
            rgb_data = np.stack((np.zeros_like(rgb_data), np.zeros_like(rgb_data), rgb_data), axis=2)
        else:
            return

        # Normalize the RGB data
        rgb_data = (rgb_data - np.min(rgb_data)) / (np.max(rgb_data) - np.min(rgb_data))

        # Display the RGB image
        plt.figure(figsize=(8, 8))
        plt.imshow(rgb_data)
        plt.axis('off')
        plt.show()

    def export_channel(self):
        if self.data is None or self.selected_channel is None:
            return

        channel_index = self.selected_channel
        selected_band = self.data.read_band(channel_index)

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Channel", "", "TIFF Files (*.tif)")
        if save_path:
            # Save the selected channel as a TIFF file
            tifffile.imwrite(save_path, selected_band)

    def select_roi(self):
        if self.data is None or self.selected_channel is None:
            return

        channel_index = self.selected_channel
        selected_band = self.data.read_band(channel_index)

        # Display the image
        fig, ax = plt.subplots()
        ax.imshow(selected_band, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Select ROI for Channel {channel_index + 1}")

        # Enable interactive mode
        plt.ion()

        # Select the ROI interactively
        roi = plt.ginput(2, timeout=0)

        # Disable interactive mode
        plt.ioff()

        plt.close(fig)

        if len(roi) == 2:
            # Extract the ROI coordinates
            x1, y1 = int(roi[0][0]), int(roi[0][1])
            x2, y2 = int(roi[1][0]), int(roi[1][1])

            # Crop the ROI
            roi_data = selected_band[y1:y2, x1:x2]

            # Store the ROI data
            self.roi_data = roi_data

            # Display the ROI
            fig, ax = plt.subplots()
            ax.imshow(roi_data, cmap='gray')
            ax.axis('off')
            ax.set_title(f"ROI for Channel {channel_index + 1}")

            # Set different x-axis limits
            ax.set_xlim(0, roi_data.shape[1])  # Adjust the limits according to your data

            plt.show()

            # Compute the spectral signature
            spectral_signature = np.mean(roi_data, axis=0)

            # Plot the spectral signature
            plt.figure()
            plt.plot(spectral_signature)
            plt.xlabel("Band")
            plt.ylabel("Reflectance")
            plt.title("Spectral Signature")
            plt.show()


app = QApplication([])
window = HyperspectralTool()
window.show()
app.exec_()
