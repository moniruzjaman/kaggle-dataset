"""Print the signature, size and label count of any ONNX model file.

Useful before wiring a new model into the app, or to find out what an unknown
.onnx file in models/ actually is:

    python inspect_onnx.py models/crop_leaf_diseases_vit.onnx

Only onnxruntime is required, so this works with the lightweight runtime.
"""

import argparse
import json
import os
from glob import glob

import onnxruntime as ort

# Every .onnx file listed with its side-car weights and the folder that holds it
DEFAULT_PATTERN = os.path.join("models", "*.onnx")


def side_car_files(model_path):
    """Return (path, size) for the model plus any *.onnx.data side-car files."""
    files = [(model_path, os.path.getsize(model_path))]
    for extra in sorted(glob(model_path + ".data")):
        files.append((extra, os.path.getsize(extra)))
    return files


def describe(model_path, expected_classes=None):
    """Load model_path and print its inputs, outputs, size and class count."""
    total = sum(size for _path, size in side_car_files(model_path))
    print("[INFO] {}".format(model_path))
    print("[INFO]   size: {:.2f} MB (with side-car weights)".format(
        total / (1024.0 * 1024.0)))

    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as error:  # noqa: BLE001 - report any load failure to the user
        print("[ERROR]   could not load: {}".format(error))
        return None

    print("[INFO]   inputs:")
    for value in session.get_inputs():
        print("[INFO]     {} {} {}".format(value.name, value.shape, value.type))
    print("[INFO]   outputs:")
    for value in session.get_outputs():
        print("[INFO]     {} {} {}".format(value.name, value.shape, value.type))

    # The classifier head size is the last dimension of the first output
    output_shape = session.get_outputs()[0].shape
    class_count = output_shape[-1] if isinstance(output_shape[-1], int) else None
    if class_count is None:
        print("[INFO]   class count: dynamic")
        return None
    print("[INFO]   class count: {}".format(class_count))

    label_path = os.path.join(os.path.dirname(model_path), "class_names.json")
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as handle:
            labels = json.load(handle)
        if len(labels) == class_count:
            print("[INFO]   class_names.json: matches ({} labels)".format(len(labels)))
        else:
            print("[WARN]   class_names.json has {} labels but the model outputs {} "
                  "- predictions would not line up".format(len(labels), class_count))
    else:
        print("[WARN]   no class_names.json next to this model, so label order is "
              "unknown to the app")
    if expected_classes is not None and class_count != expected_classes:
        print("[WARN]   this model has {} classes, the app's PlantVillage model has {}".format(
            class_count, expected_classes))
    return class_count


def main():
    parser = argparse.ArgumentParser(
        description="Describe one or more ONNX model files.")
    parser.add_argument("models", nargs="*", default=None,
                        help="model paths (default: every models/*.onnx)")
    parser.add_argument("--expected-classes", type=int, default=None,
                        help="warn when a model does not output this many classes")
    args = parser.parse_args()

    paths = args.models or sorted(glob(DEFAULT_PATTERN))
    if not paths:
        raise SystemExit("[ERROR] no .onnx files found (looked for {})".format(
            DEFAULT_PATTERN))

    for index, model_path in enumerate(paths):
        if index:
            print()
        if not os.path.exists(model_path):
            print("[ERROR] {} not found".format(model_path))
            continue
        describe(model_path, args.expected_classes)


if __name__ == "__main__":
    main()
