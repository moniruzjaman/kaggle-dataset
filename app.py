from flask import Flask, request, render_template
import os
import sys

# Lightweight inference helper (onnxruntime + numpy + Pillow, no TensorFlow)
from predict import load_predictor

app = Flask(__name__)

# Directory where uploaded leaves are stored
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

# Load the pretrained PlantVillage model once at startup
try:
    predictor = load_predictor()
except FileNotFoundError as error:
    print("[ERROR] {}".format(error))
    sys.exit(1)


def model_predict(img_path, predictor):
    """Return the predicted class name and its confidence for an image."""
    label, confidence, _ = predictor.predict_top(img_path)
    return label, confidence


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(file_path)

        # Make prediction
        pred_class, confidence = model_predict(file_path, predictor)

        result = {
            'class': pred_class,
            'confidence': '{:.2f}%'.format(confidence * 100.0)
        }
        return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
