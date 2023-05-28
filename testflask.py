import numpy as np
import matplotlib.pyplot as plt
from spectral import *
from flask import Flask, render_template, request
import os

app = Flask(__name__)

class HyperspectralTool:
    def __init__(self):
        self.image_path = ""
        self.selected_channel = None
        self.export_path = ""
        self.num_channels = 0

    def load_image(self, image_path):
        self.image_path = image_path
        self.selected_channel = None

        hyper_image = open_image(self.image_path)
        self.num_channels = hyper_image.shape[2]

    def select_channel(self, index):
        self.selected_channel = index

    def display_channel(self):
        if self.image_path and self.selected_channel is not None:
            hyper_image = open_image(self.image_path)
            selected_band = hyper_image.read_band(self.selected_channel)

            plt.imshow(selected_band, cmap='gray')
            plt.title("Channel {}".format(self.selected_channel))
            plt.savefig("static/channel_plot.png")
            plt.close()

    def convert_to_rgb(self):
        if self.image_path:
            hyper_image = open_image(self.image_path)
            rgb_image = hyper_image.rgb()

            plt.imshow(rgb_image)
            plt.title("RGB Image")
            plt.savefig("static/rgb_image.png")
            plt.close()

    def export_channel(self):
        if self.image_path and self.selected_channel is not None:
            hyper_image = open_image(self.image_path)
            selected_band = hyper_image.read_band(self.selected_channel)

            export_image_path = os.path.join("static", "exported_channel.png")
            export_image = selected_band.astype(np.uint8)
            plt.imsave(export_image_path, export_image, cmap='gray')
            return export_image_path

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
            plt.savefig("static/spectral_signatures.png")
            plt.close()


tool = HyperspectralTool()  # Create an instance of HyperspectralTool
@app.route('/')
def index():
    return render_template('index.html', tool=tool)  # Pass the instance as a context variable

if __name__ == '__main__':
    app.run()


@app.route('/load_image', methods=['POST'])
def load_image():
    image_path = request.form.get('image_path')
    tool.load_image(image_path)
    return ""

@app.route('/select_channel', methods=['POST'])
def select_channel():
    index = int(request.form.get('channel_index'))
    tool.select_channel(index)
    return ""

@app.route('/display_channel', methods=['POST'])
def display_channel():
    tool.display_channel()
    return ""

@app.route('/convert_to_rgb', methods=['POST'])
def convert_to_rgb():
    tool.convert_to_rgb()
    return ""

@app.route('/export_channel', methods=['POST'])
def export_channel():
    export_image_path = tool.export_channel()
    return export_image_path

@app.route('/select_roi', methods=['POST'])
def select_roi():
    tool.select_roi()
    return ""

# if __name__ == '__main__':
#     tool = HyperspectralTool()
#     app.run()
