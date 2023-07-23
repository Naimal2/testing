import matplotlib.pyplot as plt
from spectral import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QComboBox, QPushButton, \
    QFileDialog, QHBoxLayout, QSplitter
from PyQt5.QtGui import QPixmap
import tifffile
import os
import imageio
from scipy.io import loadmat
import numpy as np
from matplotlib.patches import Rectangle
spectral.settings.envi_support_nonlowercase_params = True
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import models, layers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Cursor
from sklearn.preprocessing import LabelEncoder
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
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
        self.img_loaded = None
        self.applied_illumination = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model_results = {}
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

        # Select ROI button
        self.select_roi_button = QPushButton("Select ROI", self)
        self.select_roi_button.clicked.connect(self.select_roi)

        # Load dataset button
        self.load_dataset_button = QPushButton("Load Dataset", self)
        self.load_dataset_button.clicked.connect(self.load_dataset_dialog)

        # Apply model button
        self.apply_model_button = QPushButton("Apply Model", self)
        self.apply_model_button.clicked.connect(self.apply_selected_model)

        # Dropdown list for model selection
        self.model_selection_label = QLabel("Select Model:")
        self.model_selection_combo = QComboBox(self)
        self.model_selection_combo.addItems(
            ["SVM",  "KNN", "Random Forest", "Decision Tree"]
        )


        # Create a button to display the model results
        self.show_results_button = QPushButton("Show Model Results", self)
        self.show_results_button.clicked.connect(self.show_model_results)

        # Results label
        self.result_label = QLabel(self)
        self.result_label.setText("Results will be displayed here.")

        # Result figure and canvas
        self.result_figure = Figure(figsize=(8, 5))
        self.result_canvas = FigureCanvas(self.result_figure)

        # Layout for left pane (toolkits)
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
        layout.addWidget(self.select_roi_button)
        layout.addWidget(self.load_dataset_button)
        layout.addWidget(self.model_selection_label)
        layout.addWidget(self.model_selection_combo)
        layout.addWidget(self.apply_model_button)
        layout.addWidget(self.show_results_button)
        layout.addStretch()  # Add stretch to push widgets to the top

        # Create a left pane widget and set its layout
        left_pane_widget = QWidget(self)
        left_pane_widget.setLayout(layout)

        # Layout for right pane (results)
        right_pane_layout = QVBoxLayout()
        right_pane_layout.addWidget(self.result_label)
        right_pane_layout.addWidget(self.result_canvas)

        # Create a right pane widget and set its layout
        right_pane_widget = QWidget(self)
        right_pane_widget.setLayout(right_pane_layout)

        # Create a QSplitter to divide the main window into two sections (left and right)
        splitter = QSplitter(self)
        splitter.addWidget(left_pane_widget)
        splitter.addWidget(right_pane_widget)

        self.setCentralWidget(splitter)

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

        # Clear the previous figure
        self.result_figure.clear()

        # Add a subplot to the figure
        ax = self.result_figure.add_subplot(111)

        # Display the image in the right pane
        ax.imshow(normalized_band, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Channel {channel_index + 1}")

        # Update the canvas to show the new figure
        self.result_canvas.draw()


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

        # Clear the previous figure
        self.result_figure.clear()

        # Add subplots to the figure
        ax1 = self.result_figure.add_subplot(121)
        ax2 = self.result_figure.add_subplot(122)

        # Plot the unnormalized reflectance graph
        ax1.plot(wavelengths, reflectance)
        ax1.set_xlabel('wavelength, nm')
        ax1.set_ylabel('unnormalized reflectance')
        ax1.set_title('Unnormalized Reflectance')

        # Plot the normalized reflectance graph
        ax2.plot(wavelengths, normalized_reflectance)
        ax2.set_xlabel('wavelength, nm')
        ax2.set_ylabel('normalized reflectance')
        ax2.set_title('Normalized Reflectance')

        # Adjust the layout to prevent overlap
        self.result_figure.tight_layout()

        # Update the canvas to show the new figure
        self.result_canvas.draw()

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

                # Clear the previous figure
                self.result_figure.clear()

                # Add a subplot to the figure
                ax = self.result_figure.add_subplot(111)

                # Plot the Reflected Radiance Spectrum graph
                ax.plot(wavelengths, radiance, 'b', label='Applied Illumination')
                ax.set_xlabel('wavelength, nm')
                ax.set_ylabel('radiance, arbitrary units')
                ax.set_title('Reflected Radiance Spectrum')
                ax.legend()

                # Update the canvas to show the new figure
                self.result_canvas.draw()

                self.applied_illumination = illumination  # Store the applied illumination


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
                XYZ = XYZ / np.max(XYZ)

                # Convert XYZ to sRGB
                RGB = XYZ2sRGB_exgamma(XYZ)

                # Clip RGB values
                RGB = np.clip(RGB, 0, 1)

                # Perform R, G, and B operations
                R = RGB[:, :, 0] * 0.5
                G = RGB[:, :, 1] * 2.0
                B = RGB[:, :, 2] * 1.5

                # Normalize pixel values to the range [0, 1]
                R = R / np.max(R)
                G = G / np.max(G)
                B = B / np.max(B)

                # Clear the previous figure
                self.result_figure.clear()

                # Add subplots to the figure
                ax_r = self.result_figure.add_subplot(131)
                ax_g = self.result_figure.add_subplot(132)
                ax_b = self.result_figure.add_subplot(133)

                # Display R, G, and B channels
                ax_r.imshow(R, cmap='gray', aspect='auto', origin='lower')
                ax_r.set_title('R Channel')
                ax_r.axis('off')

                ax_g.imshow(G, cmap='gray', aspect='auto', origin='lower')
                ax_g.set_title('G Channel')
                ax_g.axis('off')

                ax_b.imshow(B, cmap='gray', aspect='auto', origin='lower')
                ax_b.set_title('B Channel')
                ax_b.axis('off')

                # Update the canvas to show the new figure
                self.result_canvas.draw()

                # Display the RGB image
                fig_rgb, ax_rgb = plt.subplots(figsize=(6, 6))
                ax_rgb.imshow(RGB)
                ax_rgb.set_title('RGB Image')
                ax_rgb.axis('off')

                # Show the RGB image in a separate window (optional)
                plt.show()

                self.applied_illumination = illumination  # Store the applied illumination


    def select_roi(self):
        if self.data is None or self.selected_channel is None:
            return

        if self.applied_illumination is None:  # Check if illumination is applied
            print("Please apply an illumination spectrum.")
            return

        illumination = self.applied_illumination  # Use the applied illumination

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



            # Crop the ROI
            roi_data = selected_band[y1:y2, x1:x2]

            # Check if the ROI data contains any NaN values
            if np.isnan(roi_data).any():
                print("Selected ROI contains NaN values. Please try again.")
                return

            # Store the ROI data
            self.roi_data = roi_data
            self.roi_coordinates = (x1, y1, x2, y2)

            # Compute the spectral signature
            spectral_signature = np.mean(roi_data, axis=0)

            # Define the wavelengths based on the length of spectral_signature
            wavelengths = np.linspace(x1, x2, len(spectral_signature))

            # Store the spectral signature
            self.roi_spectral_signature = (wavelengths, spectral_signature)

            # Clear the previous figure
            self.result_figure.clear()

            # Add subplots to the figure
            ax_roi = self.result_figure.add_subplot(121)
            ax_signature = self.result_figure.add_subplot(122)

            # Display the ROI
            ax_roi.imshow(selected_band, cmap='gray')
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='r', facecolor='none')
            ax_roi.add_patch(rect)
            ax_roi.axis('off')
            ax_roi.set_title(f"ROI for Channel {channel_index + 1}")

            # Display the spectral signature
            wavelengths, spectral_signature = self.roi_spectral_signature
            ax_signature.plot(wavelengths, spectral_signature)
            ax_signature.set_xlabel("Wavelength (nm)")
            ax_signature.set_ylabel("Reflectance")
            ax_signature.set_title("Spectral Signature")

            # Adjust the layout to prevent overlap
            self.result_figure.tight_layout()

            # Update the canvas to show the new figure
            self.result_canvas.draw()

            plt.show()
        else:
            # Clear the ROI data if ROI selection was canceled
            self.roi_data = None
            self.roi_coordinates = None
            self.roi_spectral_signature = None

    def load_dataset_dialog(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("CSV Files (*.csv)")
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths and file_paths[0]:
                file_path = file_paths[0]
                self.load_dataset_from_csv(file_path)
            else:
                print("No file selected.")
        else:
            print("File dialog canceled.")

    def load_dataset_from_csv(self, file_path):
        if file_path:
            alldata = pd.read_csv(file_path)

            # Check if the DataFrame has at least 225 columns
            if alldata.shape[1] >= 225:
                X = alldata.iloc[:, 1:224].values
                y = alldata.iloc[:, 224].values

                # Use LabelEncoder on the combined dataset (X and y) to ensure consistent encoding
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)

                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_encoded, test_size=0.20,
                                                                                        random_state=0)

                sc = StandardScaler()
                self.X_train = sc.fit_transform(self.X_train)
                self.X_test = sc.transform(self.X_test)
                print("Dataset loaded successfully.")
            else:
                print("The DataFrame does not have enough columns for X and y.")
        else:
            print("Invalid file path or file path not selected.")

    def apply_selected_model(self):
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            print("Please load the dataset first (Option 1).")
            return

        selected_model = self.model_selection_combo.currentText()

        if selected_model == "SVM":
            model = SVC(kernel='poly', random_state=0)
        elif selected_model == "KNN":
            model = KNeighborsClassifier(n_neighbors=3)
        elif selected_model == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif selected_model == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        else:
            print("Invalid model selected.")
            return

        # Fit the model on the training data
        model.fit(self.X_train, self.y_train)

        # Make predictions on the test data
        y_pred_encoded = model.predict(self.X_test)

        # For non-binary classification models, convert the predictions back to original labels
        label_encoder = LabelEncoder()

        # Fit the label encoder on the combined dataset (y_train and y_test) to ensure consistent encoding
        combined_y = np.concatenate((self.y_train, self.y_test))
        label_encoder.fit(combined_y)

        y_pred = label_encoder.inverse_transform(y_pred_encoded)

        # Calculate accuracy, precision, and recall
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=1)  # Set zero_division=1

        print(f"Accuracy of {selected_model}: {accuracy:.2f}")
        print(f"Precision of {selected_model}: {precision:.2f}")
        print(f"Recall of {selected_model}: {recall:.2f}")

        # Store the results in the model_results dictionary
        self.model_results[selected_model] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }

        # Update the result label to display the current metrics
        result_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}"
        self.result_label.setText(result_text)

    def show_model_results(self):
        if not self.model_results:
            print("No model results to display. Please apply a model first (Option 2).")
            return

        models = list(self.model_results.keys())
        accuracies = [result["accuracy"] for result in self.model_results.values()]
        precisions = [result["precision"] for result in self.model_results.values()]
        recalls = [result["recall"] for result in self.model_results.values()]

        # Plot bar charts for accuracy, precision, and recall
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(models, accuracies)
        ax.set_xlabel("Models")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Comparison - Accuracy")
        plt.xticks(rotation=45)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(models, precisions)
        ax.set_xlabel("Models")
        ax.set_ylabel("Precision")
        ax.set_title("Model Comparison - Precision")
        plt.xticks(rotation=45)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(models, recalls)
        ax.set_xlabel("Models")
        ax.set_ylabel("Recall")
        ax.set_title("Model Comparison - Recall")
        plt.xticks(rotation=45)

        plt.show()
app = QApplication([])
window = HyperspectralTool()
window.show()
app.exec_()