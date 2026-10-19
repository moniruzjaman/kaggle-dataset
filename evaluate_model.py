"""Check the pretrained model against a local PlantVillage folder.

The script streams one image at a time, so it stays lightweight even though the
dataset contains tens of thousands of images. Extract "kaggle dataset.zip" at the
project root first (it creates plant-village/PlantVillage/<class>/...):

    python evaluate_model.py --per-class 20
"""

import argparse
import os
import random

from dataset_paths import DATASET_CANDIDATES, DEFAULT_DATA_DIR, resolve_data_dir
from predict import DEFAULT_MODEL_DIR, load_predictor

# Image formats used by the PlantVillage dataset (.JPG) and common alternatives
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def collect_samples(data_dir, per_class, seed):
    """Return [(image_path, class_name), ...] with at most per_class per folder."""
    if not os.path.isdir(data_dir):
        raise SystemExit(
            "[ERROR] dataset folder not found: {}\n"
            "        extract 'kaggle dataset.zip' at the project root first.".format(data_dir))

    samples = []
    class_names = sorted(name for name in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, name)))
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        files = [name for name in os.listdir(class_dir)
                 if name.lower().endswith(IMAGE_SUFFIXES)]
        random.Random(seed).shuffle(files)
        if per_class > 0:
            files = files[:per_class]
        for file_name in files:
            samples.append((os.path.join(class_dir, file_name), class_name))
    return class_names, samples


def main():
    parser = argparse.ArgumentParser(
        description="Measure the accuracy of the pretrained model on local images.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="folder containing one sub-folder per class "
                             "(auto-detects {} when the default is missing)".format(
                                 ", ".join(DATASET_CANDIDATES)))
    parser.add_argument("--per-class", type=int, default=20,
                        help="images sampled per class (0 = all images)")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed")
    parser.add_argument("--show-misses", type=int, default=10,
                        help="how many misclassifications to print")
    parser.add_argument("--model-dir", default=None,
                        help="folder holding model.onnx + class_names.json, e.g. the "
                             "int8 build 'models/plantvillage_mobilenetv3_int8' "
                             "(default: {})".format(os.path.relpath(DEFAULT_MODEL_DIR)))
    args = parser.parse_args()

    class_names, samples = collect_samples(
        resolve_data_dir(args.data_dir), args.per_class, args.seed)
    if not samples:
        raise SystemExit("[ERROR] no images found in {}".format(args.data_dir))

    if args.model_dir:
        predictor = load_predictor(
            model_path=os.path.join(args.model_dir, "model.onnx"),
            class_names_path=os.path.join(args.model_dir, "class_names.json"))
    else:
        predictor = load_predictor()
    print("[INFO] evaluating {} images from {} classes".format(len(samples), len(class_names)))

    correct = 0
    per_class_total = {name: 0 for name in class_names}
    per_class_correct = {name: 0 for name in class_names}
    misses = []

    for index, (image_path, expected) in enumerate(samples, start=1):
        try:
            predicted, confidence, _ = predictor.predict_top(image_path)
        except (OSError, ValueError) as error:
            print("[WARN] skipping {} ({})".format(image_path, error))
            continue
        per_class_total[expected] += 1
        if predicted == expected:
            correct += 1
            per_class_correct[expected] += 1
        elif len(misses) < args.show_misses:
            misses.append((image_path, expected, predicted, confidence))
        if index % 25 == 0 or index == len(samples):
            print("[INFO] {}/{} processed (accuracy {:.2f}%)".format(
                index, len(samples), 100.0 * correct / index))

    evaluated = sum(per_class_total.values())
    print("[RESULT] accuracy: {:.2f}% ({}/{})".format(
        100.0 * correct / evaluated, correct, evaluated))
    for class_name in class_names:
        total = per_class_total[class_name]
        if total:
            print("[RESULT] {:<45} {}/{}".format(
                class_name, per_class_correct[class_name], total))
    for image_path, expected, predicted, confidence in misses:
        print("[MISS] {} -> {} ({:.2f}%) expected {}".format(
            image_path, predicted, confidence * 100.0, expected))


if __name__ == "__main__":
    main()
