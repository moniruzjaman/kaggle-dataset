"""Train a small PlantVillage classifier and export it for the lightweight app.

Differences from the first version of this script:

* images are streamed with tf.data instead of being loaded into RAM (the old
  version needed well over 10 GB for the full dataset)
* the dataset folder is auto-detected via dataset_paths.py, so it works whether
  "kaggle dataset.zip" was extracted to the project root or into plant-village/
* --per-class caps how many images per class are used, for quick runs
* --export-onnx writes model.onnx + class_names.json with exactly the interface
  the app already uses: NCHW float32, ImageNet-normalised, sorted class order and
  raw logits as output. predict.py therefore needs no change:

      $env:PLANT_MODEL_DIR = "models\\plantvillage_trained"
      python predict.py --image path\\to\\leaf.jpg

Quick run (a few minutes on CPU):

    python train_model.py --per-class 200 --img-size 160 --epochs 3 --export-onnx

Full run:

    python train_model.py --epochs 20 --export-onnx

Needs requirements-train.txt (TensorFlow, scikit-learn, matplotlib, tf2onnx).
"""

import argparse
import json
import os
import random
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from dataset_paths import class_names_in, resolve_data_dir

# Normalisation applied to every image, identical to predict.py so the exported
# ONNX model is a drop-in replacement for the one downloaded from Hugging Face
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Ratio used to build the validation split
VAL_SPLIT = 0.2

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def collect_split(data_dir, class_names, per_class, val_split, seed):
    """Return (train_pairs, val_pairs) of (image path, class index) split per class."""
    rng = random.Random(seed)
    train_pairs = []
    val_pairs = []

    for index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        files = [name for name in os.listdir(class_dir)
                 if name.lower().endswith(IMAGE_SUFFIXES)]
        rng.shuffle(files)
        if per_class > 0:
            files = files[:per_class]
        if not files:
            continue
        val_count = max(1, int(round(len(files) * val_split)))
        val_count = min(val_count, len(files) - 1) if len(files) > 1 else 0
        for offset, file_name in enumerate(files):
            pair = (os.path.join(class_dir, file_name), index)
            (val_pairs if offset < val_count else train_pairs).append(pair)

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    return train_pairs, val_pairs


def preprocess_image(path, label, class_count, img_size, training):
    """Decode, resize (shorter side then centre crop), decode label and normalise.

    The resize and normalise steps mirror predict.py exactly, so a model trained
    here consumes the same input tensor the app already produces.
    """
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)  # 0..1

    # Equivalent of torchvision.transforms.Resize(round(256/224 * img_size)) on
    # the shorter side, followed by a centre crop of img_size
    short_size = int(round(img_size * 256.0 / 224.0))
    shape = tf.shape(image)[:2]
    scale = tf.cast(short_size, tf.float32) / tf.cast(tf.reduce_min(shape), tf.float32)
    new_size = tf.cast(tf.round(tf.cast(shape, tf.float32) * scale), tf.int32)
    image = tf.image.resize(image, new_size, method="bilinear", antialias=True)
    image = tf.image.resize_with_crop_or_pad(image, img_size, img_size)

    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.rot90(image, k=tf.random.uniform((), 0, 4, dtype=tf.int32))
        # Brightness/contrast jitter, then scale back to 0..1 if it drifted
        image = tf.clip_by_value(
            tf.image.random_brightness(image, 0.1), 0.0, 1.0)

    image = (image - tf.constant(MEAN, tf.float32)) / tf.constant(STD, tf.float32)
    # NCHW because that is what predict.py feeds and what the export exposes
    image = tf.transpose(image, (2, 0, 1))
    return image, tf.one_hot(label, class_count)


def make_dataset(pairs, class_count, img_size, batch_size, training, seed):
    """Build a prefetching tf.data pipeline, so images are never all in RAM."""
    paths = [path for path, _label in pairs]
    labels = [label for _path, label in pairs]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(pairs), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: preprocess_image(path, label, class_count, img_size, training),
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(base_name, img_size, class_count, alpha):
    """Transfer-learning model: frozen ImageNet base + small classification head.

    The NCHW input and the absent preprocessing layer are deliberate: images
    arrive already normalised (see preprocess_image) and the head has no softmax,
    so the exported graph returns logits exactly like the model shipped with the
    app, which keeps the confidence values in predict.py meaningful.
    """
    if base_name == "mobilenetv3":
        base = tf.keras.applications.MobileNetV3Small(
            input_shape=(img_size, img_size, 3), alpha=alpha, include_top=False,
            weights="imagenet", include_preprocessing=False)
    else:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3), include_top=False,
            weights="imagenet")
    base.trainable = False

    # NCHW tensor (already normalised) -> NHWC for the Keras base
    inputs = tf.keras.Input(shape=(3, img_size, img_size), name="input")
    x = tf.keras.layers.Permute((2, 3, 1), name="to_nhwc")(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="pool")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Dense(class_count, name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="plantvillage_" + base_name), base


def compile_model(model, learning_rate):
    """Compile with a logits loss, matching the softmax-free head."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])
    return model


def unfreeze_for_fine_tuning(model, base, fraction):
    """Unfreeze the top `fraction` of the base, keeping batchnorm layers frozen."""
    base.trainable = True
    cutoff = int(len(base.layers) * (1.0 - fraction))
    for index, layer in enumerate(base.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = index >= cutoff
    return model


def evaluate_on_split(model, dataset, class_names):
    """Print a scikit-learn classification report for the validation split."""
    probabilities = model.predict(dataset, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    expected = np.concatenate([np.argmax(batch_labels.numpy(), axis=1)
                               for _batch_images, batch_labels in dataset])
    print(classification_report(expected, predicted, target_names=class_names,
                                digits=4, zero_division=0))
    return float(np.mean(predicted == expected))


def plot_history(history, output_path):
    """Save the training curves, same idea as the original script."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")
    plt.figure()
    for metric in ("loss", "accuracy"):
        if metric in history.history:
            plt.plot(history.history[metric], label="train_" + metric)
            plt.plot(history.history["val_" + metric], label="val_" + metric)
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(output_path)
    plt.close()
    print("[INFO] wrote {}".format(output_path))


def export_onnx(model, output_dir, img_size, class_names, opset):
    """Write model.onnx + class_names.json that predict.py can load as-is."""
    import tf2onnx

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "model.onnx")
    signature = [tf.TensorSpec([None, 3, img_size, img_size], tf.float32, name="input")]
    tf2onnx.convert.from_keras(model, input_signature=signature,
                               opset=opset, output_path=onnx_path)

    class_names_path = os.path.join(output_dir, "class_names.json")
    with open(class_names_path, "w", encoding="utf-8") as handle:
        json.dump(class_names, handle, indent=2)
        handle.write("\n")

    print("[INFO] exported {}".format(onnx_path))
    print("[INFO] wrote {} ({} labels)".format(class_names_path, len(class_names)))
    return onnx_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a PlantVillage classifier and optionally export ONNX.")
    parser.add_argument("--data-dir", default=None,
                        help="dataset folder with one sub-folder per class "
                             "(auto-detected by default)")
    parser.add_argument("--per-class", type=int, default=0,
                        help="images per class, 0 = every image (slow on CPU)")
    parser.add_argument("--img-size", type=int, default=224,
                        help="input size; smaller trains and runs faster")
    parser.add_argument("--base", choices=("mobilenetv3", "mobilenetv2"),
                        default="mobilenetv3",
                        help="ImageNet backbone used for transfer learning")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="width multiplier for the MobileNetV3 base "
                             "(1.0, 0.75 or 0.5)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="epochs training the head with a frozen base")
    parser.add_argument("--fine-tune-epochs", type=int, default=0,
                        help="extra epochs with the top of the base unfrozen")
    parser.add_argument("--fine-tune-fraction", type=float, default=0.25,
                        help="fraction of the base layers unfrozen for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="plant_disease_model.keras",
                        help="where to save the trained Keras model")
    parser.add_argument("--plot", default="plot.png",
                        help="where to save the training curves")
    parser.add_argument("--export-onnx", action="store_true",
                        help="also write an ONNX model for predict.py")
    parser.add_argument("--onnx-dir",
                        default=os.path.join("models", "plantvillage_trained"),
                        help="folder for model.onnx + class_names.json")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    class_names = class_names_in(data_dir)
    if not class_names:
        raise SystemExit(
            "[ERROR] no class folders found in {}.\n"
            "        extract 'kaggle dataset.zip' at the project root first.".format(
                data_dir))

    train_pairs, val_pairs = collect_split(
        data_dir, class_names, args.per_class, VAL_SPLIT, args.seed)
    if not train_pairs:
        raise SystemExit("[ERROR] no training images found in {}".format(data_dir))

    print("[INFO] dataset: {}".format(data_dir))
    print("[INFO] images: {} train / {} validation from {} classes".format(
        len(train_pairs), len(val_pairs), len(class_names)))
    print("[INFO] base: {} alpha {} at {}x{}".format(
        args.base, args.alpha, args.img_size, args.img_size))

    train_ds = make_dataset(train_pairs, len(class_names), args.img_size,
                            args.batch_size, True, args.seed)
    val_ds = make_dataset(val_pairs, len(class_names), args.img_size,
                          args.batch_size, False, args.seed)

    model, base = build_model(args.base, args.img_size, len(class_names), args.alpha)
    compile_model(model, args.learning_rate)
    print("[INFO] training head for {} epochs...".format(args.epochs))
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        verbose=2)
    print("[INFO] head training took {:.1f} s".format(time.time() - start))

    if args.fine_tune_epochs > 0:
        unfreeze_for_fine_tuning(model, base, args.fine_tune_fraction)
        compile_model(model, args.learning_rate / 10.0)
        print("[INFO] fine-tuning for {} epochs...".format(args.fine_tune_epochs))
        start = time.time()
        more = model.fit(train_ds, validation_data=val_ds, verbose=2,
                         initial_epoch=args.epochs,
                         epochs=args.epochs + args.fine_tune_epochs)
        for key, values in more.history.items():
            history.history.setdefault(key, []).extend(values)
        print("[INFO] fine-tuning took {:.1f} s".format(time.time() - start))

    print("[INFO] evaluating on the validation split...")
    accuracy = evaluate_on_split(model, val_ds, class_names)
    print("[RESULT] validation accuracy: {:.2f}%".format(accuracy * 100.0))

    model.save(args.output)
    print("[INFO] saved {}".format(args.output))
    plot_history(history, args.plot)

    if args.export_onnx:
        export_onnx(model, args.onnx_dir, args.img_size, class_names, args.opset)
        print("[INFO] use it with: $env:PLANT_MODEL_DIR = \"{}\"; "
              "python predict.py --image path\\to\\leaf.jpg".format(args.onnx_dir))


if __name__ == "__main__":
    main()
