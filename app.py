import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# NEW: Import Hugging Face downloader
from huggingface_hub import hf_hub_download  # NEW

app = Flask(__name__)

# Dynamically determine path to models folder relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATHS = {
    "prototxt": os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt"),
    "caffemodel": os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel"),
    "npy": os.path.join(MODEL_DIR, "pts_in_hull.npy")
}

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# NEW: Download .caffemodel if not present
def download_model():  # NEW
    if not os.path.exists(MODEL_PATHS["caffemodel"]):
        print("Downloading model from Hugging Face...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_file = hf_hub_download(
            repo_id="KailashNaidu07/image-colorization-model",
            filename="colorization_release_v2.caffemodel"
        )
        os.rename(model_file, MODEL_PATHS["caffemodel"])
        print("Model downloaded successfully.")  # NEW

download_model()  # NEW

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    try:
        pts_in_hull = np.load(MODEL_PATHS["npy"])
        net = cv2.dnn.readNetFromCaffe(MODEL_PATHS["prototxt"], MODEL_PATHS["caffemodel"])

        # Set cluster centers
        pts = pts_in_hull.transpose().reshape(2, 313, 1, 1)
        net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full((1, 313), 2.606, dtype="float32")]

        return net
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None

net = load_model()
MODEL_LOADED = net is not None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if not MODEL_LOADED:
        return render_template('error.html', error="Model failed to load. Check paths or model files.")

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('index.html', error='No file selected')

        if file and allowed_file(file.filename):
            try:
                in_memory_file = file.read()
                npimg = np.frombuffer(in_memory_file, np.uint8)
                img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

                if img is None:
                    return render_template('index.html', error='Invalid image file')

                # Convert to LAB and process L channel
                lab_img = cv2.cvtColor(img.astype("float32") / 255.0, cv2.COLOR_BGR2LAB)
                L = cv2.resize(cv2.split(lab_img)[0], (224, 224))
                L -= 50

                net.setInput(cv2.dnn.blobFromImage(L))
                ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
                ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

                colorized = np.concatenate([cv2.split(lab_img)[0][:, :, np.newaxis], ab_channel], axis=2)
                colorized = np.clip(cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR), 0, 1)
                colorized = (255 * colorized).astype("uint8")

                # Encode colorized image to base64
                _, buffer = cv2.imencode('.png', colorized)
                encoded_image = base64.b64encode(buffer).decode('utf-8')

                # Encode grayscale version to base64
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                _, gray_buf = cv2.imencode('.png', gray_bgr)
                gray_base64 = base64.b64encode(gray_buf).decode('utf-8')

                return render_template('result.html', image_data=encoded_image, gray_data=gray_base64)

            except Exception as e:
                return render_template('index.html', error=f'Processing error: {str(e)}')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
