"""Download the lightweight pretrained PlantVillage disease classifier.

The model is MobileNetV3-Small (1.53M parameters, ~6.5 MB ONNX export) fine-tuned
on the same 15-class PlantVillage dataset that ships in "kaggle dataset.zip", so
there is no need to train anything. Only the Python standard library is used
here, which means the download works without TensorFlow, PyTorch or pip
installing anything heavy.
"""

import argparse
import json
import os
import urllib.request

# Hugging Face repository that hosts the pretrained model
MODEL_REPO = "imaflower/plantvillage-mobilenetv3"
BASE_URL = "https://huggingface.co/{}/resolve/main".format(MODEL_REPO)

# Local directory that will hold the downloaded model
MODEL_DIR = os.path.join("models", "plantvillage_mobilenetv3")

# Files needed at inference time ("model.onnx" and "model.onnx.data" must stay
# together because the ONNX graph stores its weights in the side-car file)
MODEL_FILES = ("model.onnx", "model.onnx.data", "class_names.json")

# The classifier is trained on the 15-class PlantVillage subset
EXPECTED_CLASSES = 15


def download_file(url, destination, force=False):
    """Stream a single file to disk and report its size."""
    if os.path.exists(destination) and not force:
        size = os.path.getsize(destination)
        if size > 0:
            print("[INFO] {} already present ({:.2f} MB), skipping".format(
                os.path.basename(destination), size / (1024 * 1024)))
            return size

    request = urllib.request.Request(url, headers={"User-Agent": "plantvillage-lite/1.0"})
    print("[INFO] downloading {}...".format(url))
    with urllib.request.urlopen(request) as response, open(destination, "wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)

    size = os.path.getsize(destination)
    print("[INFO] saved {} ({:.2f} MB)".format(
        os.path.basename(destination), size / (1024 * 1024)))
    return size


def main():
    parser = argparse.ArgumentParser(
        description="Download the pretrained lightweight PlantVillage model (ONNX).")
    parser.add_argument("--output-dir", default=MODEL_DIR,
                        help="directory that will receive the model files")
    parser.add_argument("--force", action="store_true",
                        help="re-download files even if they already exist")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_bytes = 0
    for file_name in MODEL_FILES:
        url = "{}/{}".format(BASE_URL, file_name)
        destination = os.path.join(args.output_dir, file_name)
        total_bytes += download_file(url, destination, force=args.force)

    # Sanity check the label file so a broken download fails loudly
    class_names_path = os.path.join(args.output_dir, "class_names.json")
    with open(class_names_path, "r", encoding="utf-8") as handle:
        class_names = json.load(handle)
    if len(class_names) != EXPECTED_CLASSES:
        raise SystemExit("[ERROR] expected {} classes but found {} in class_names.json".format(
            EXPECTED_CLASSES, len(class_names)))

    print("[INFO] model: {} ({} classes)".format(MODEL_REPO, len(class_names)))
    print("[INFO] total size: {:.2f} MB".format(total_bytes / (1024 * 1024)))
    print("[INFO] saved to: {}".format(os.path.abspath(args.output_dir)))
    print("[INFO] next step: python predict.py --image path\\to\\leaf.jpg")


if __name__ == "__main__":
    main()
