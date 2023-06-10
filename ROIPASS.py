import matplotlib.pyplot as plt
from spectral import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QComboBox, QPushButton, \
    QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap
import tifffile
import os
import imageio
from scipy.io import loadmat
import numpy as np
from matplotlib.patches import Rectangle
spectral.settings.envi_support_nonlowercase_params = True
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.widgets import RectangleSelector
from mpldatacursor import datacursor
from matplotlib.patches import Rectangle

spectral.settings.envi_support_nonlowercase_params = True
import csv

def XYZ2sRGB_exgamma(XYZ):
    # Conversion matrix from XYZ to sRGB
    M = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])

    # Linearize the XYZ values
    XYZ_linear = XYZ ** 2.2

    # Convert XYZ to sRGB
    RGB_linear = M @ XYZ_linear[..., np.newaxis]
    RGB_linear = np.squeeze(RGB_linear)

    # Apply gamma correction to obtain sRGB values
    RGB = np.where(RGB_linear <= 0.0031308,
                   12.92 * RGB_linear,
                   1.055 * RGB_linear ** (1.0 / 2.4) - 0.055)

    return RGB


class HyperspectralTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hyperspectral Visualization Tool")

        self.image_path = ""
        self.selected_channel = None
        self.selected_illumination = None
        self.data = None
        self.roi_start = None
        self.roi_end = None
        self.roi_data = None
        # Load button
        self.load_button = QPushButton("Load Hyperspectral Image", self)
        self.load_button.clicked.connect(self.load_image)

        # Channel selection
        self.channel_label = QLabel("Select Channel:")
        self.channel_combo = QComboBox(self)
        self.channel_combo.currentIndexChanged.connect(self.select_channel)

        # Display channel button
        self.display_channel_button = QPushButton("Display Channel", self)
        self.display_channel_button.clicked.connect(self.display_channel)

        # Reflectance graph button
        self.reflectance_button = QPushButton("Show Reflectance Graph", self)
        self.reflectance_button.clicked.connect(self.show_reflectance_graph)

        # Illumination selection
        self.illumination_label = QLabel("Select Illumination Spectrum:")
        self.illumination_combo = QComboBox(self)
        self.illumination_combo.currentIndexChanged.connect(self.select_illumination)

        # Load illumination button
        self.load_illumination_button = QPushButton("Load Illumination Spectrum", self)
        self.load_illumination_button.clicked.connect(self.load_illumination)

        # Apply illumination button
        self.apply_illumination_button = QPushButton("Apply Illumination", self)
        self.apply_illumination_button.clicked.connect(self.apply_illumination)

        # RGB conversion button
        self.rgb_conversion_button = QPushButton("Convert to RGB", self)
        self.rgb_conversion_button.clicked.connect(self.convert_to_rgb)
        # ROI selection button
        self.roi_selection_button = QPushButton("Select ROI", self)
        self.roi_selection_button.clicked.connect(self.select_roi)


        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.channel_label)
        layout.addWidget(self.channel_combo)
        layout.addWidget(self.display_channel_button)
        layout.addWidget(self.reflectance_button)
        layout.addWidget(self.illumination_label)
        layout.addWidget(self.illumination_combo)
        layout.addWidget(self.load_illumination_button)
        layout.addWidget(self.apply_illumination_button)
        layout.addWidget(self.rgb_conversion_button)
        layout.addWidget(self.roi_selection_button)

        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.illumination_files = []

    def select_channel(self, index):
        self.selected_channel = index

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Supported Files (*.hdr *.tiff *.mat)")
        if file_path:
            self.image_path = file_path
            self.channel_combo.clear()

            # Load hyperspectral image
            if file_path.endswith('.hdr'):
                self.data = spectral.open_image(file_path).load()  # Using spectral library for .hdr files

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
        if self.data is None:
            return

        if self.selected_channel is None:
            print("Please select a channel.")
            return

        channel_index = self.selected_channel
        selected_band = self.data[:, :, channel_index]

        # Normalize the pixel values to the range [0, 1]
        normalized_band = selected_band.astype(float) / np.max(selected_band)

        plt.imshow(normalized_band, cmap='gray')
        plt.axis('off')
        plt.title(f"Channel {channel_index + 1}")
        plt.show()

    def get_wavelength(self):
        if self.image_path.endswith('.mat'):
            mat = loadmat(self.image_path)
            wavelength = mat['wavelength'].flatten()
        elif self.image_path.endswith('.hdr'):
            img = spectral.open_image(self.image_path)
            wavelength = img.bands.centers
        else:
            # TODO: Add support for other file formats if needed
            return None

        return wavelength
    def show_reflectance_graph(self):
        if self.data is None:
            return

        if self.selected_channel is None:
            print("Please select a channel.")
            return

        channel_index = self.selected_channel
        reflectance = self.data[141, 75, :]  # Assuming the pixel coordinates are (141, 75)

        # Normalize the reflectance values
        normalized_reflectance = reflectance / np.max(reflectance)

        # Flatten the reflectance arrays
        reflectance = reflectance.flatten()
        normalized_reflectance = normalized_reflectance.flatten()

        # Define the wavelengths array based on the reflectance length
        wavelengths = np.arange(400, 720, 10)

        # Ensure the lengths of wavelengths and reflectance match
        if len(wavelengths) > len(reflectance):
            wavelengths = wavelengths[:len(reflectance)]
        elif len(wavelengths) < len(reflectance):
            reflectance = reflectance[:len(wavelengths)]
            normalized_reflectance = normalized_reflectance[:len(wavelengths)]

        # Plot the unnormalized reflectance graph
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(wavelengths, reflectance)
        plt.xlabel('wavelength, nm')
        plt.ylabel('unnormalized reflectance')
        plt.title('Unnormalized Reflectance')

        # Plot the normalized reflectance graph
        plt.subplot(122)
        plt.plot(wavelengths, normalized_reflectance)
        plt.xlabel('wavelength, nm')
        plt.ylabel('normalized reflectance')
        plt.title('Normalized Reflectance')

        # Adjust the layout to prevent overlap
        plt.tight_layout()

        # Display the graphs
        plt.show()
        # Generate CSV file
        filename = "reflectance_data.csv"
        data = zip(wavelengths, reflectance, normalized_reflectance)
        headers = ['Wavelength (nm)', 'Unnormalized Reflectance', 'Normalized Reflectance']

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(data)

        print("CSV file generated:", filename)


    def select_illumination(self, index):
        if index < len(self.illumination_files):
            self.selected_illumination = self.illumination_files[index]
            print("Illumination spectrum selected:", self.selected_illumination)
        else:
            self.selected_illumination = None

    def load_illumination(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Illumination Spectrum", "", "Supported Files (*.mat)")
        if file_path:
            self.illumination_files.append(file_path)
            self.illumination_combo.addItem(os.path.basename(file_path))

    def apply_illumination(self):
        if self.data is None:
            return

        if not self.selected_illumination:
            print("Please select an illumination spectrum.")
            return

        illumination_data = loadmat(self.selected_illumination)

        for illum_key in illumination_data.keys():
            if illum_key.startswith('illum'):
                illumination = np.resize(illumination_data[illum_key], self.data.shape[0])

                radiances = np.zeros_like(self.data)
                for i in range(self.data.shape[-1]):
                    radiances[:, :, i] = self.data[:, :, i] * illumination[i]

                radiance = radiances[141, 75, :]
                wavelengths = np.arange(400, 730, 10)[:radiance.shape[0]]

                plt.figure()
                plt.plot(wavelengths, radiance, 'b', label='Applied Illumination')
                plt.xlabel('wavelength, nm')
                plt.ylabel('radiance, arbitrary units')
                plt.title('Reflected Radiance Spectrum')
                plt.legend()
                plt.show()



    def convert_to_rgb(self):
        if self.data is None:
            return

        if not self.selected_illumination:
            print("Please select an illumination spectrum.")
            return

        illumination_data = loadmat(self.selected_illumination)

        for illum_key in illumination_data.keys():
            if illum_key.startswith('illum'):
                illumination = np.resize(illumination_data[illum_key], self.data.shape[0])

                radiances = np.zeros_like(self.data)
                for i in range(self.data.shape[-1]):
                    radiances[:, :, i] = self.data[:, :, i] * illumination[i]

                r, c, w = radiances.shape
                radiances = radiances.reshape(r * c, w)

                # Load xyzbar
                xyzbar_data = loadmat('xyzbar.mat')
                xyzbar = xyzbar_data['xyzbar']
                # Ensure the number of columns in radiances matches the number of rows in xyzbar
                if radiances.shape[1] != xyzbar.shape[0]:
                    # Reshape xyzbar to match the number of columns in radiances
                    xyzbar = np.resize(xyzbar, (radiances.shape[1], xyzbar.shape[1]))

                # Calculate XYZ values
                XYZ = np.matmul(radiances, xyzbar)

                # Reshape XYZ values
                XYZ = XYZ.reshape(r, c, 3)

                # Normalize XYZ values
                XYZ = np.maximum(XYZ, 0)
                XYZ = XYZ / np.max(XYZ)

                # Convert XYZ to sRGB
                RGB = XYZ2sRGB_exgamma(XYZ)

                # Clip RGB values
                RGB = np.maximum(RGB, 0)
                RGB = np.minimum(RGB, 1)

                # Perform R, G, and B operations
                R = RGB[:, :, 0] * 0.5
                G = RGB[:, :, 1] * 2.0
                B = RGB[:, :, 2] * 1.5

                # Normalize pixel values to the range [0, 1]
                R = R / np.max(R)
                G = G / np.max(G)
                B = B / np.max(B)

                # Display R, G, and B channels separately
                fig, (ax_r, ax_g, ax_b) = plt.subplots(1, 3, figsize=(12, 4))

                ax_r.imshow(R, cmap='gray', aspect='auto', origin='lower')
                ax_r.set_title('R Channel')
                ax_r.axis('off')

                ax_g.imshow(G, cmap='gray', aspect='auto', origin='lower')
                ax_g.set_title('G Channel')
                ax_g.axis('off')

                ax_b.imshow(B, cmap='gray', aspect='auto', origin='lower')
                ax_b.set_title('B Channel')
                ax_b.axis('off')

                # Display the RGB image
                fig_rgb, ax_rgb = plt.subplots(figsize=(6, 6))
                ax_rgb.imshow(RGB)
                ax_rgb.set_title('RGB Image')
                ax_rgb.axis('off')

                plt.tight_layout()
                plt.show()

    def select_roi(self):
        if self.data is None or self.selected_channel is None:
            return

        channel_index = self.selected_channel
        selected_band = self.data[:, :, channel_index]

        # Display the image
        fig, ax = plt.subplots()
        ax.imshow(selected_band, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Select ROI for Channel {channel_index + 1}")

        # Enable interactive mode
        plt.ion()

        # Select the ROI interactively
        roi = plt.ginput(2, timeout=-1)

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

            # Define the wavelengths based on the length of spectral_signature
            wavelengths = np.linspace(400, 730, len(spectral_signature))

            # Plot the spectral signature with wavelengths
            plt.figure()
            plt.plot(wavelengths, spectral_signature)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Reflectance")
            plt.title("Spectral Signature")
            plt.show()
        else:
            # Clear the ROI data if ROI selection was canceled
            self.roi_data = None




app = QApplication([])
window = HyperspectralTool()
window.show()
app.exec_()
