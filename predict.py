"""Lightweight inference for the pretrained PlantVillage disease classifier.

Only onnxruntime + numpy + Pillow are required (about 30 MB of wheels), so no
TensorFlow or PyTorch install is needed. Typical usage:

    python predict.py --image path\\to\\leaf.jpg

The same class can be imported from the Flask app:

    from predict import load_predictor
    predictor = load_predictor()
    label, confidence, ranked = predictor.predict_top("leaf.jpg")
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# Float32 model that ships with the project (about 6.2 MB)
DEFAULT_MODEL_DIR = os.path.join(HERE, "models", "plantvillage_mobilenetv3")

# Set PLANT_MODEL_DIR to switch to another copy without touching any code, e.g.
# the smaller build produced by quantize_model.py or a model trained by
# train_model.py:
#     $env:PLANT_MODEL_DIR = "models\plantvillage_mobilenetv3_light"
MODEL_DIR = os.environ.get("PLANT_MODEL_DIR", DEFAULT_MODEL_DIR)
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
DEFAULT_CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

# Preprocessing used while the model was trained: resize the shorter side to 256,
# center-crop to 224x224 and normalise with the ImageNet statistics.
IMG_SIZE = 224
RESIZE_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_class_names(class_names_path=DEFAULT_CLASS_NAMES_PATH):
    """Return the ordered list of labels: index -> class name."""
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(
            "class names not found at {}. Run 'python download_pretrained_model.py' first.".format(
                class_names_path))
    with open(class_names_path, "r", encoding="utf-8") as handle:
        class_names = json.load(handle)
    return list(class_names)


def preprocess_image(image_path, img_size=IMG_SIZE, resize_size=None):
    """Load an image and return an NCHW float32 batch ready for the ONNX session.

    img_size comes from the model itself (see model_input_size), so models trained
    at other resolutions work without further changes. The shortcut of resizing
    the shorter side to img_size * 256/224 and centre cropping matches the
    preprocessing the shipped model was trained with.
    """
    if resize_size is None:
        resize_size = int(round(img_size * 256.0 / 224.0))

    image = Image.open(image_path).convert("RGB")

    # Equivalent of torchvision.transforms.Resize(256): shorter side becomes 256
    scale = resize_size / float(min(image.size))
    if scale != 1.0:
        new_size = (max(1, int(round(image.width * scale))),
                    max(1, int(round(image.height * scale))))
        image = image.resize(new_size, Image.BILINEAR)

    # Centre crop to img_size (the shipped model used torchvision CenterCrop(224))
    left = (image.width - img_size) // 2
    top = (image.height - img_size) // 2
    image = image.crop((left, top, left + img_size, top + img_size))

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return np.transpose(array, (2, 0, 1))[np.newaxis, ...].astype(np.float32)


def model_input_size(session):
    """Return (img_size, resize_size) implied by the session's input shape.

    Falls back to the 224 default when the graph declares dynamic dimensions.
    """
    shape = session.get_inputs()[0].shape
    if len(shape) == 4:
        height, width = shape[2], shape[3]
        if (isinstance(height, int) and isinstance(width, int)
                and height == width and height > 0):
            img_size = int(height)
            return img_size, int(round(img_size * 256.0 / 224.0))
    return IMG_SIZE, RESIZE_SIZE


def softmax(logits):
    """Numerically stable softmax over a 1-D array of scores."""
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


class PlantDiseasePredictor:
    """Thin wrapper around the ONNX model that returns ranked predictions."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH,
                 class_names_path=DEFAULT_CLASS_NAMES_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "model not found at {}. Run 'python download_pretrained_model.py' first.".format(
                    model_path))
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        # Read the resolution from the graph so any trained size works
        self.img_size, self.resize_size = model_input_size(self.session)
        self.class_names = load_class_names(class_names_path)

    def predict(self, image_path, top_k=3):
        """Return [(label, probability), ...] sorted from most to least likely."""
        batch = preprocess_image(image_path, self.img_size, self.resize_size)
        scores = self.session.run(None, {self.input_name: batch})[0][0]
        probabilities = softmax(scores)
        top_k = max(1, min(top_k, len(self.class_names)))
        best = np.argsort(probabilities)[::-1][:top_k]
        return [(self.class_names[int(index)], float(probabilities[int(index)])) for index in best]

    def predict_top(self, image_path):
        """Return (label, confidence, ranked_list) for the most likely class."""
        ranked = self.predict(image_path, top_k=len(self.class_names))
        label, confidence = ranked[0]
        return label, confidence, ranked


def load_predictor(model_path=DEFAULT_MODEL_PATH,
                   class_names_path=DEFAULT_CLASS_NAMES_PATH):
    """Convenience factory used by the web app."""
    return PlantDiseasePredictor(model_path=model_path, class_names_path=class_names_path)


def main():
    parser = argparse.ArgumentParser(
        description="Classify a plant leaf image with the pretrained PlantVillage model.")
    parser.add_argument("--image", required=True, help="path to a leaf image")
    parser.add_argument("--top-k", type=int, default=3, help="how many predictions to print")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="path to model.onnx")
    parser.add_argument("--class-names", default=DEFAULT_CLASS_NAMES_PATH,
                        help="path to class_names.json")
    args = parser.parse_args()

    predictor = load_predictor(model_path=args.model, class_names_path=args.class_names)
    label, confidence, ranked = predictor.predict_top(args.image)

    print("[INFO] image: {}".format(args.image))
    print("[INFO] input size: {}x{}".format(predictor.img_size, predictor.img_size))
    for index, (name, probability) in enumerate(ranked[:args.top_k], start=1):
        print("[RESULT] {}. {} - {:.2f}%".format(index, name, probability * 100.0))
    print("[RESULT] predicted: {} ({:.2f}%)".format(label, confidence * 100.0))


if __name__ == "__main__":
    main()
