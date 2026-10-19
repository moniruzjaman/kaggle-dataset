"""Build a smaller copy of the pretrained PlantVillage model and verify it.

The shipped model is float32 and 6.2 MB. Two of its layers (the classifier's
Gemm/MatMul pair) hold 2.4 MB of that, and quantizing only those to int8
measured a 29% smaller file with identical predictions:

    python quantize_model.py                 # -> models/plantvillage_mobilenetv3_light

Full int8 quantization of every layer was tried as well (both weights-only and
QDQ with image calibration): it shrank the file by about 70% but collapsed
accuracy from 99% to 5-19% and got slower, so the script deletes such a build and
exits with status 1 instead of leaving it for the app to pick up.

Other modes:

    python quantize_model.py --mode fp16     # half size, ~15x slower on CPU
    python quantize_model.py --mode static   # full int8 QDQ (rejected here)
    python quantize_model.py --mode dynamic  # full int8 weights-only (rejected)

Use a converted model with:

    $env:PLANT_MODEL_DIR = "models\\plantvillage_mobilenetv3_light"
    python predict.py --image path\\to\\leaf.jpg

Needs requirements-quantize.txt (onnx, onnxruntime, numpy).
"""

import argparse
import json
import os
import shutil
import statistics
import tempfile
import time

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_dynamic,
                                      quantize_static)

from evaluate_model import DEFAULT_DATA_DIR, collect_samples, resolve_data_dir
from predict import DEFAULT_MODEL_DIR, PlantDiseasePredictor, preprocess_image

# The original float32 model is always the source here, independent of
# PLANT_MODEL_DIR, so re-running stays idempotent.
SOURCE_MODEL_DIR = DEFAULT_MODEL_DIR
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIRS = {
    "gemm": os.path.join(HERE, "models", "plantvillage_mobilenetv3_light"),
    "fp16": os.path.join(HERE, "models", "plantvillage_mobilenetv3_fp16"),
    "static": os.path.join(HERE, "models", "plantvillage_mobilenetv3_int8"),
    "dynamic": os.path.join(HERE, "models", "plantvillage_mobilenetv3_int8_dynamic"),
}

# Operator types used by the recommended "gemm" preset: the classifier fully
# connected layers hold 2.4 MB of the 6.2 MB model, and quantizing only those
# measured a 29% smaller file with unchanged predictions and slightly faster
# inference.
GEMM_PRESET_OPS = ["MatMul", "Gemm"]

WEIGHT_TYPES = {"quint8": QuantType.QUInt8, "qint8": QuantType.QInt8}
CALIBRATION_METHODS = {"minmax": CalibrationMethod.MinMax,
                       "entropy": CalibrationMethod.Entropy,
                       "percentile": CalibrationMethod.Percentile}


def folder_size(path):
    """Total size in bytes of every file inside a folder."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for file_name in files:
            total += os.path.getsize(os.path.join(root, file_name))
    return total


def describe_size(path):
    """Return (human readable size, byte size) for a file or folder."""
    size = folder_size(path) if os.path.isdir(path) else os.path.getsize(path)
    return "{:.2f} MB".format(size / (1024.0 * 1024.0)), size


def raw_scores(predictor, image_path):
    """Run the session on one image and return (argmax index, logits)."""
    batch = preprocess_image(image_path, predictor.img_size, predictor.resize_size)
    logits = predictor.session.run(None, {predictor.input_name: batch})[0][0]
    return int(np.argmax(logits)), logits


def median_latency_ms(predictor, batch, runs, warmup=3):
    """Median wall-clock time in milliseconds for a single forward pass."""
    for _ in range(warmup):
        predictor.session.run(None, {predictor.input_name: batch})
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        predictor.session.run(None, {predictor.input_name: batch})
        timings.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(timings)


def sanitize_model(source_model, work_dir):
    """Drop optional value_info annotations that break ONNX shape inference.

    The hosted export (PyTorch 2.10, opset 18) annotates a few activations with
    shapes that contradict what ONNX infers, and onnxruntime.quantization runs
    shape inference before quantizing, so it aborts with "Inferred shape and
    existing shape differ". value_info is optional metadata, so removing it is
    safe: the quantizer re-infers everything it needs from the weights.
    """
    model = onnx.load(source_model)  # also pulls in the model.onnx.data side-car
    removed = len(model.graph.value_info)
    del model.graph.value_info[:]
    target = os.path.join(work_dir, "sanitized.onnx")
    onnx.save_model(model, target)
    return target, removed


def build_quantized_model(source_model, output_model, weight_type, per_channel,
                          op_types):
    """Quantize source_model into output_model and return the written bytes."""
    side_car = output_model + ".data"
    if os.path.exists(side_car):
        os.remove(side_car)

    print("[INFO] quantizing {} as {}".format(
        os.path.basename(source_model), weight_type))
    quantize_dynamic(
        model_input=source_model,
        model_output=output_model,
        weight_type=WEIGHT_TYPES[weight_type],
        per_channel=per_channel,
        op_types_to_quantize=op_types,
    )

    written = os.path.getsize(output_model)
    if os.path.exists(side_car):
        written += os.path.getsize(side_car)
    return written

class ImageCalibrationReader(CalibrationDataReader):
    """Feeds real leaf images to the static quantizer, one batch per call."""

    def __init__(self, image_paths, predictor):
        self.input_name = predictor.input_name
        # Preprocess exactly like the model under test expects
        self._batches = (preprocess_image(path, predictor.img_size, predictor.resize_size)
                         for path in image_paths)

    def get_next(self):
        try:
            return {self.input_name: next(self._batches)}
        except StopIteration:
            return None


def build_static_quantized_model(source_model, output_model, image_paths,
                                 predictor, per_channel, weight_type, op_types,
                                 calibrate_method):
    """QDQ quantization with MinMax calibration; this is what the conv layers need.

    Dynamic quantization of Conv produces ConvInteger, which measured both
    inaccurate (5% accuracy vs 99%) and several times slower on the CPU
    provider for this model, so conv weights are quantized statically instead,
    using real leaf images to calibrate the activation ranges.
    """
    side_car = output_model + ".data"
    if os.path.exists(side_car):
        os.remove(side_car)

    print("[INFO] calibrating and quantizing {} on {} images ({} method)".format(
        os.path.basename(source_model), len(image_paths), calibrate_method))
    quantize_static(
        model_input=source_model,
        model_output=output_model,
        calibration_data_reader=ImageCalibrationReader(image_paths, predictor),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=WEIGHT_TYPES[weight_type],
        per_channel=per_channel,
        calibrate_method=CALIBRATION_METHODS[calibrate_method],
        op_types_to_quantize=op_types,
    )

    written = os.path.getsize(output_model)
    if os.path.exists(side_car):
        written += os.path.getsize(side_car)
    return written


def build_fp16_model(source_model, output_model):
    """Halve the file size by storing every weight as float16.

    The graph inputs and outputs stay float32 (keep_io_types), so predict.py
    needs no change. float16 was the only conversion that survived the accuracy
    check for this export; int8 Conv quantization measured 5-19% accuracy.
    """
    try:
        from onnxconverter_common import float16
    except ImportError:
        raise SystemExit(
            "[ERROR] float16 conversion needs onnxconverter-common. Install it "
            "with: uv pip install --python .venv\\Scripts\\python.exe "
            "onnxconverter-common")

    model = onnx.load(source_model)
    del model.graph.value_info[:]
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save_model(converted, output_model)

    side_car = output_model + ".data"
    written = os.path.getsize(output_model)
    if os.path.exists(side_car):
        written += os.path.getsize(side_car)
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Quantize the pretrained PlantVillage model to int8 and compare "
                    "it against the float32 original.")
    parser.add_argument("--source-dir", default=SOURCE_MODEL_DIR,
                        help="folder with the float32 model.onnx + class_names.json")
    parser.add_argument("--output-dir", default=None,
                        help="folder that will receive the converted model "
                             "(default: models/plantvillage_mobilenetv3_<mode>)")
    parser.add_argument("--data-dir", default=None,
                        help="dataset used for the comparison (auto-detected by default)")
    parser.add_argument("--per-class", type=int, default=10,
                        help="images per class used for the comparison")
    parser.add_argument("--runs", type=int, default=30,
                        help="forward passes used to measure latency")
    parser.add_argument("--weight-type", choices=sorted(WEIGHT_TYPES), default="quint8",
                        help="int8 weight format used by ONNX Runtime")
    parser.add_argument("--per-channel", action="store_true",
                        help="quantize per output channel (slower to build, often "
                             "more accurate)")
    parser.add_argument("--op-types", nargs="*", default=None,
                        help="restrict dynamic quantization to these operator types, "
                             "e.g. MatMul Gemm")
    parser.add_argument("--mode", choices=("gemm", "fp16", "static", "dynamic"),
                        default="gemm",
                        help="gemm (default) = int8 weights for the classifier only, "
                             "verified as accurate and faster; fp16 = half size but "
                             "about 15x slower on CPU; static/dynamic = full int8, "
                             "which measured unusable for this export and is rejected "
                             "by the check below")
    parser.add_argument("--calib-method", choices=sorted(CALIBRATION_METHODS),
                        default="minmax",
                        help="activation range estimation used by the static mode")
    parser.add_argument("--calib-per-class", type=int, default=5,
                        help="images per class used to calibrate the static mode")
    parser.add_argument("--min-agreement", type=float, default=98.0,
                        help="reject the int8 build when it agrees with the float32 "
                             "model on fewer than this percentage of test images")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIRS[args.mode]

    source_model = os.path.join(args.source_dir, "model.onnx")
    source_class_names = os.path.join(args.source_dir, "class_names.json")
    if not os.path.exists(source_model):
        raise SystemExit(
            "[ERROR] float32 model not found at {}. Run "
            "'python download_pretrained_model.py' first.".format(source_model))

    os.makedirs(args.output_dir, exist_ok=True)
    output_model = os.path.join(args.output_dir, "model.onnx")
    data_dir = resolve_data_dir(args.data_dir or DEFAULT_DATA_DIR)

    # Reference model: also reveals the input name and serves as the baseline
    float_predictor = PlantDiseasePredictor(
        model_path=source_model, class_names_path=source_class_names)

    # model.onnx + model.onnx.data count as one model
    original_size = describe_size(args.source_dir)
    with tempfile.TemporaryDirectory() as work_dir:
        prepared_model, removed = sanitize_model(source_model, work_dir)
        if removed:
            print("[INFO] stripped {} stale shape annotations that block ONNX "
                  "shape inference".format(removed))
        if args.mode == "fp16":
            quantized_bytes = build_fp16_model(prepared_model, output_model)
        elif args.mode == "gemm":
            quantized_bytes = build_quantized_model(
                prepared_model, output_model, args.weight_type, False,
                args.op_types or GEMM_PRESET_OPS)
        elif args.mode == "static":
            _names, calibration = collect_samples(data_dir, args.calib_per_class, seed=7)
            if not calibration:
                raise SystemExit(
                    "[ERROR] no calibration images found in {}. Use '--mode dynamic' "
                    "to quantize without the dataset.".format(data_dir))
            quantized_bytes = build_static_quantized_model(
                prepared_model, output_model, [path for path, _name in calibration],
                float_predictor, args.per_channel, args.weight_type,
                args.op_types, args.calib_method)
        else:
            quantized_bytes = build_quantized_model(
                prepared_model, output_model, args.weight_type, args.per_channel,
                args.op_types)

    class_names_path = os.path.join(args.output_dir, "class_names.json")
    shutil.copyfile(source_class_names, class_names_path)

    converted_predictor = PlantDiseasePredictor(
        model_path=output_model, class_names_path=class_names_path)
    label = "fp16" if args.mode == "fp16" else "int8"
    print("[INFO] float32:   {} labels from {}".format(
        len(float_predictor.class_names), os.path.relpath(source_model)))
    print("[INFO] {}: {} labels from {}".format(
        label.ljust(7), len(converted_predictor.class_names),
        os.path.relpath(output_model)))

    # Compare both models on the same real images
    _class_names, samples = collect_samples(data_dir, args.per_class, seed=42)
    if not samples:
        raise SystemExit("[ERROR] no images found in {} to compare against".format(data_dir))

    agreement = 0
    float_correct = 0
    converted_correct = 0
    evaluated = 0
    disagreements = []

    for image_path, expected in samples:
        try:
            float_index, _ = raw_scores(float_predictor, image_path)
            converted_index, _ = raw_scores(converted_predictor, image_path)
        except (OSError, ValueError) as error:
            print("[WARN] skipping {} ({})".format(image_path, error))
            continue

        evaluated += 1
        float_label = float_predictor.class_names[float_index]
        converted_label = converted_predictor.class_names[converted_index]
        agreement += int(float_label == converted_label)
        float_correct += int(float_label == expected)
        converted_correct += int(converted_label == expected)
        if float_label != converted_label and len(disagreements) < 10:
            disagreements.append((image_path, expected, float_label, converted_label))
    batch = preprocess_image(samples[0][0], float_predictor.img_size,
                             float_predictor.resize_size)
    float_ms = median_latency_ms(float_predictor, batch, args.runs)
    converted_ms = median_latency_ms(converted_predictor, batch, args.runs)

    print("[RESULT] images compared: {}".format(evaluated))
    print("[RESULT] float32 accuracy: {:.2f}% ({}/{})".format(
        100.0 * float_correct / evaluated, float_correct, evaluated))
    print("[RESULT] {:<7} accuracy: {:.2f}% ({}/{})".format(
        label, 100.0 * converted_correct / evaluated, converted_correct, evaluated))
    print("[RESULT] top-1 agreement: {:.2f}% ({}/{})".format(
        100.0 * agreement / evaluated, agreement, evaluated))
    print("[RESULT] latency: float32 {:.2f} ms -> {} {:.2f} ms ({:+.0f}%)".format(
        float_ms, label, converted_ms,
        100.0 * (converted_ms - float_ms) / float_ms))
    print("[RESULT] size: {} -> {} ({:.0f}% smaller)".format(
        original_size[0], describe_size(args.output_dir)[0],
        100.0 * (1.0 - float(quantized_bytes) / float(original_size[1]))))
    for image_path, expected, float_label, converted_label in disagreements:
        print("[DIFF] {} expected {} | float32 {} | {} {}".format(
            image_path, expected, float_label, label, converted_label))

    agreement_percent = 100.0 * agreement / evaluated
    accuracy_drop = 100.0 * (float_correct - converted_correct) / evaluated
    if agreement_percent < args.min_agreement or accuracy_drop > 1.0:
        # Never leave a misbehaving build behind for the app to pick up
        for path in (output_model, class_names_path):
            if os.path.exists(path):
                os.remove(path)
        print("[RESULT] verdict: rejected ({:.2f}% agreement, {:+.2f} points "
              "accuracy) - kept the float32 model".format(
                  agreement_percent, -accuracy_drop))
        raise SystemExit(1)

    print("[RESULT] verdict: usable ({:.2f}% agreement, {:+.2f} points "
          "accuracy)".format(agreement_percent, -accuracy_drop))
    print("[INFO] to use it: $env:PLANT_MODEL_DIR = \"{}\"; "
          "python predict.py --image path\\to\\leaf.jpg".format(args.output_dir))


if __name__ == "__main__":
    main()

