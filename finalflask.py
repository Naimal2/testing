from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from spectral import *
import tifffile
import os

app = Flask(__name__)

spectral.settings.envi_support_nonlowercase_params = True
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for matplotlib to avoid GUI dependencies

@app.route('/')
def index():
    return render_template('flask.html')

@app.route('/load_image', methods=['POST'])
def load_image():
    file_path = request.files['file']
    if file_path:
        file_path.save('temp.hdr')
        # Load hyperspectral image
        data = open_image('temp.hdr')

        # Get the number of channels
        num_channels = data.shape[-1]

        return render_template('flask.html', num_channels=num_channels)
    else:
        return render_template('flask.html', error_message='Failed to load the image.')

@app.route('/display_channel', methods=['POST'])
def display_channel():
    selected_channel = int(request.form['channel'])
    data = open_image('temp.hdr')
    selected_band = data.read_band(selected_channel)

    # Normalize the pixel values to the range [0, 1]
    normalized_band = selected_band.astype(float) / np.max(selected_band)

    plt.imshow(normalized_band, cmap='gray')
    plt.axis('off')
    plt.title(f"Channel {selected_channel}")
    plt.savefig('static/channel_image.png')  # Save the figure as a static image
    plt.close()

    return render_template('display.html')

if __name__ == '__main__':
    app.run(debug=True)
