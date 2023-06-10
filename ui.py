import matplotlib.pyplot as plt
from spectral import *
from flask import Flask, render_template, request, send_file
import tifffile
import os
import imageio
from scipy.io import loadmat
import numpy as np
import csv
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_image', methods=['POST'])
def load_image():
    image_file = request.files['image']
    if image_file:
        image_path = os.path.join('uploads', image_file.filename)
        image_file.save(image_path)

        channel_combo = []
        num_channels = 0

        if image_path.endswith('.hdr'):
            data = spectral.open_image(image_path).load()
            num_channels = data.shape[-1]
            channel_combo = [str(i) for i in range(1, num_channels + 1)]
        elif image_path.endswith('.tiff'):
            data = plt.imread(image_path)
            num_channels = 1
            channel_combo = ['1']
        elif image_path.endswith('.mat'):
            mat_data = loadmat(image_path)
            data = mat_data['reflectances']
            num_channels = data.shape[-1]
            channel_combo = [str(i) for i in range(1, num_channels + 1)]
        else:
            return "Unsupported file type."

        return render_template('index.html', image_path=image_path, channel_combo=channel_combo)

@app.route('/display_channel', methods=['POST'])
def display_channel():
    image_path = request.form['image_path']
    selected_channel = int(request.form['selected_channel'])

    if image_path.endswith('.hdr'):
        data = spectral.open_image(image_path).load()
    elif image_path.endswith('.tiff'):
        data = plt.imread(image_path)
    elif image_path.endswith('.mat'):
        mat_data = loadmat(image_path)
        data = mat_data['reflectances']
    else:
        return "Unsupported file type."

    selected_band = data[:, :, selected_channel - 1]
    normalized_band = selected_band.astype(float) / np.max(selected_band)

    plt.imshow(normalized_band, cmap='gray')
    plt.axis('off')
    plt.title(f"Channel {selected_channel}")

    # Save the plot as an image file
    image_buffer = BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)

    # Return the image file to the client
    # return send_file(image_buffer, mimetype='image/png')
    # Save the plot as an image file
    image_buffer = BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)

    # Encode the image data as Base64
    encoded_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

    # Return the encoded image data to the client
    return encoded_image

@app.route('/show_reflectance_graph', methods=['POST'])
def show_reflectance_graph():
    image_path = request.form['image_path']
    selected_channel = int(request.form['selected_channel'])

    if image_path.endswith('.hdr'):
        data = spectral.open_image(image_path).load()
    elif image_path.endswith('.tiff'):
        data = plt.imread(image_path)
    elif image_path.endswith('.mat'):
        mat_data = loadmat(image_path)
        data = mat_data['reflectances']
    else:
        return "Unsupported file type."

    reflectance = data[141, 75, :]  # Assuming the pixel coordinates are (141, 75)
    normalized_reflectance = reflectance / np.max(reflectance)

    wavelengths = np.arange(400, 720, 10)

    if len(wavelengths) > len(reflectance):
        wavelengths = wavelengths[:len(reflectance)]
    elif len(reflectance) > len(wavelengths):
        reflectance = reflectance[:len(wavelengths)]
        normalized_reflectance = normalized_reflectance[:len(wavelengths)]

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(wavelengths, reflectance)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Reflectance Spectrum')

    plt.subplot(122)
    plt.plot(wavelengths, normalized_reflectance)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Reflectance')
    plt.title('Normalized Reflectance Spectrum')

    plt.tight_layout()

    # Save the plot as an image file
    image_buffer = BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)

    # Return the image file to the client
    # return send_file(image_buffer, mimetype='image/png')
     # Save the plot as an image file
    image_buffer = BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)

    # Encode the image data as Base64
    encoded_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

    # Return the encoded image data to the client
    return encoded_image

if __name__ == '__main__':
    app.run()
