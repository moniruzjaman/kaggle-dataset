# Plant Disease Detection System

This project detects plant diseases from leaf images. It ships with a small
pretrained CNN (MobileNetV3-Small, ~6 MB) fine-tuned on the PlantVillage
dataset, so **no training step is required** and neither TensorFlow nor PyTorch
has to be installed - inference runs on ONNX Runtime on the CPU.

## Quick start (pretrained, lightweight)

1. Create an environment and install the light runtime (about 30 MB of wheels):

   ```powershell
   uv venv --python 3.13 .venv
   uv pip install --python .venv\Scripts\python.exe -r requirements.txt
   ```

   Plain `pip` works the same way (Python 3.9-3.13 recommended; the
   `onnxruntime` wheel also resolves for Python 3.14):

   ```powershell
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```

2. Download the pretrained model (standard library only, 6.2 MB total):

   ```powershell
   .venv\Scripts\python download_pretrained_model.py
   ```

3. Classify a leaf image from the command line:

   ```powershell
   .venv\Scripts\python predict.py --image path\to\leaf.jpg --top-k 3
   ```

4. Run the Flask web application:

   ```powershell
   .venv\Scripts\python app.py
   ```

   Then open `http://localhost:5000`, select a leaf image and click
   "Detect Disease".

## Deployed version

A live instance is running on Render:

[https://plant-disease-detection.onrender.com](https://plant-disease-detection.onrender.com)

*(Replace the link above with your own Render URL after deploying — see the steps below.)*

### Deploy to Render

1. Push this repo to GitHub.
2. In the Render dashboard, create a **New Web Service** and connect the repository.
3. Render will auto-detect `render.yaml` (Python runtime, `gunicorn app:app` start command).
   The build command already installs the runtime and downloads the pretrained model.
4. Deploy — Render runs the build command, which installs the runtime and downloads
   the pretrained model (6.2 MB) from Hugging Face before the service starts.

### Test the deployed link

Open the Render URL, upload a leaf image, and click **"Detect Disease"**. The result
page shows the predicted class and confidence percentage, matching the local CLI output.

## Files

| File | Purpose |
|---|---|
| `requirements.txt` | Lightweight runtime: Flask + ONNX Runtime inference |
| `requirements-train.txt` | Extra packages, only needed to re-train from scratch |
| `requirements-quantize.txt` | Extra packages, only needed to rebuild the smaller models |
| `download_pretrained_model.py` | Downloads the pretrained model (standard library only) |
| `predict.py` | Prediction CLI plus the `load_predictor()` helper used by the app |
| `evaluate_model.py` | Measures accuracy against a local dataset folder |
| `quantize_model.py` | Builds a smaller model and verifies it against the float32 original |
| `inspect_onnx.py` | Prints inputs, outputs, size and class count of any `.onnx` file |
| `check_quantization_support.py` | Checks whether int8 quantization is reliable in this environment |
| `benchmark_models.py` | Interleaved size + latency comparison of every model in `models/` |
| `dataset_paths.py` | Finds the dataset folder (shared by the scripts, no runtime deps) |
| `app.py` | Flask web application |
| `train_model.py` | Optional transfer-learning script (TensorFlow, streams the dataset) |
| `templates/index.html`, `templates/result.html` | Web app pages |

## Pretrained model

| Item | Value |
|---|---|
| Source | Hugging Face `imaflower/plantvillage-mobilenetv3` (MIT licence) |
| Architecture | MobileNetV3-Small, 1.53M parameters, ImageNet-pretrained backbone |
| Download size | 6.16 MB (`model.onnx` + `model.onnx.data` + `class_names.json`) |
| Input | RGB, shorter side resized to 256, center-crop 224x224, ImageNet normalisation |
| Output | 15 PlantVillage classes (order taken from `class_names.json`) |
| Verified accuracy | 99.56% (224/225) and 100% (75/75) on two independent samples from this repository's dataset |
| Runtime | ONNX Runtime CPU - no GPU, no TensorFlow, no PyTorch |

Files are stored in `models/plantvillage_mobilenetv3/`. `predict.py` reads the
label order from `class_names.json` at start-up, so predictions always follow the
model's own output order instead of a hard-coded list.

`evaluate_model.py` samples images per class and prints per-class accuracy:

```powershell
.venv\Scripts\python evaluate_model.py --per-class 20   # 20 images per class
.venv\Scripts\python evaluate_model.py --per-class 0    # every image (slow)
```

The dataset folder is auto-detected (`plant-village/PlantVillage`, then
`PlantVillage/PlantVillage`, then `PlantVillage`); pass `--data-dir` to point at
another layout.

`inspect_onnx.py` prints the input/output signature, size and class count of any
ONNX file, and warns when a `class_names.json` next to it does not line up:

```powershell
.venv\Scripts\python inspect_onnx.py models\crop_leaf_diseases_vit.onnx
```

## Lighter models (optional)

`quantize_model.py` builds a smaller copy of the same model and **verifies it
against the float32 original** on real images before accepting it. A build that
disagrees, or that loses more than one accuracy point, is deleted and the script
exits with status 1, so a bad model can never silently reach the app.

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-quantize.txt
.venv\Scripts\python quantize_model.py                  # recommended build
.venv\Scripts\python quantize_model.py --mode fp16      # half size, much slower
```

Measured on this machine with `--per-class 15` (225 images) and an interleaved
latency benchmark:

| Build | Size | Accuracy | Latency | Command |
|---|---|---|---|---|
| float32 (shipped) | 6.16 MB | 99.56% | 8.24 ms | `download_pretrained_model.py` |
| **classifier int8 (`--mode gemm`, default)** | **4.35 MB** | **99.56%** | 8.11 ms | `quantize_model.py` |
| full int8, weights-only (`--mode dynamic`) | 1.69 MB | 5.33% | 22 ms | rejected |
| full int8 QDQ (`--mode static`) | 1.93 MB | 14.67% | 9.6 ms | rejected |
| float16 (`--mode fp16`) | 3.17 MB | 99.56% | 80-130 ms | `--mode fp16` |

The int8 builds are ~70% smaller but unusable here: full int8 quantization of
this export measured 5-19% accuracy even with per-channel weights and QDQ
calibration, and it is slower as well. That is not a broken environment - 
`check_quantization_support.py` builds tiny conv and depthwise-conv models,
quantizes them the same way and gets 20/20 top-1 agreement - and it reproduced
with a freshly trained TensorFlow MobileNetV3 export too, so it is specific to
these MobileNet graphs. The default build therefore quantizes only the
classifier's `Gemm`/`MatMul` layers - the two fully connected layers account for
2.4 MB of the 6.16 MB file - which measured a 100% top-1 match with the float32
model. Single-image latency fluctuates a lot between runs (roughly 4-11 ms for
both float32 and the light build); an interleaved benchmark showed them within 1%
of each other, so the win here is **file size, not speed**. float16 halves the
file size at identical accuracy but makes CPU inference roughly 15-20x slower, so
it only makes sense when disk size beats latency.

For a genuinely smaller model the useful lever is the architecture itself, not
quantization. `train_model.py --per-class 300 --img-size 128 --epochs 10
--fine-tune-epochs 3 --alpha 0.75 --export-onnx` produced
`models\plantvillage_trained\` (MobileNetV3-Small alpha 0.75 at 128x128):

| Model | Size | Accuracy | Latency (median) |
|---|---|---|---|
| shipped float32, 224x224 | 6.15 MB | 99.56% | 6.4-7.0 ms |
| `models\plantvillage_trained`, 128x128 | **2.29 MB** | 91.00% (273/300) / 88.85% val | 5-11 ms |
| `models\crop_leaf_diseases_vit.onnx` | 21.27 MB | not usable (13 classes) | 120-140 ms |

It is 2.7x smaller and trained in about 7 minutes on CPU from 4,352 images
(300 per class), so accuracy trails the shipped model, which was trained on all
20,639 images. More images, more epochs and a larger `--img-size` close that gap.
`benchmark_models.py` reproduces the latency column:

```powershell
.venv\Scripts\python benchmark_models.py
```

Attempting int8 on that trained export failed the same way as on the shipped one
(8.67% top-1 agreement, build deleted by `quantize_model.py`), which confirms the
int8 problem is specific to MobileNet-family graphs in this ONNX Runtime version
rather than a property of the downloaded weights.

Point the app or the CLI at a converted model with `PLANT_MODEL_DIR` (or
`evaluate_model.py --model-dir`):

```powershell
$env:PLANT_MODEL_DIR = "models\plantvillage_mobilenetv3_light"
.venv\Scripts\python predict.py --image path\to\leaf.jpg
.venv\Scripts\python evaluate_model.py --model-dir models\plantvillage_mobilenetv3_light
```

### About `models\crop_leaf_diseases_vit.onnx`

That file is **not** used by this project. It is a 21.27 MB Vision Transformer
exposing a `pixel_values` input and a **13-class** output, and it has no
`class_names.json`, so its label order is unknown and its classes do not match
the 15 PlantVillage folders here. Measured in the same benchmark it is also 3.5x
larger and 6.6x slower than the MobileNetV3 model the app uses (75 ms vs 11 ms
per image):

```powershell
.venv\Scripts\python inspect_onnx.py models\crop_leaf_diseases_vit.onnx
```

## Dataset

`kaggle dataset.zip` contains the 15-class PlantVillage set (20,639 images, the
same data the pretrained model was trained on). Extract it before evaluating or
re-training:

```powershell
Expand-Archive -Path "kaggle dataset.zip" -DestinationPath .
```

Resulting layout (auto-detected by `dataset_paths.py`, so extracting to the
project root or into `plant-village/` both work):

```
plant-village/
  PlantVillage/
    Pepper__bell___Bacterial_spot/
    Pepper__bell___healthy/
    Potato___Early_blight/
    ...
```

### Classes

Pepper__bell___Bacterial_spot, Pepper__bell___healthy, Potato___Early_blight,
Potato___Late_blight, Potato___healthy, Tomato_Bacterial_spot,
Tomato_Early_blight, Tomato_Late_blight, Tomato_Leaf_Mold,
Tomato_Septoria_leaf_spot, Tomato_Spider_mites_Two_spotted_spider_mite,
Tomato__Target_Spot, Tomato__Tomato_YellowLeaf__Curl_Virus,
Tomato__Tomato_mosaic_virus, Tomato_healthy

## Optional: training your own model

The web app does not need this. Keep TensorFlow out of the app runtime venv:

```powershell
uv venv --python 3.12 .venv-train
uv pip install --python .venv-train\Scripts\python.exe -r requirements-train.txt
```

Then train (the dataset folder is auto-detected, images are streamed with
`tf.data` so RAM stays flat):

```powershell
# quick run, a few minutes on CPU
.venv-train\Scripts\python train_model.py --per-class 200 --img-size 128 --epochs 5

# longer run with fine-tuning of the top of the backbone
.venv-train\Scripts\python train_model.py --per-class 300 --img-size 128 --epochs 10 `
    --fine-tune-epochs 3 --fine-tune-fraction 0.3 --alpha 0.75 --export-onnx
```

Useful flags: `--per-class` (0 = every image), `--img-size`, `--base`
(`mobilenetv3` or `mobilenetv2`), `--alpha` (backbone width), `--epochs`,
`--fine-tune-epochs`, `--batch-size`, `--learning-rate`, `--seed`, `--output`,
`--plot`, `--export-onnx`, `--onnx-dir`, `--opset`. It writes the Keras model,
`plot.png` and a scikit-learn classification report.

### What `--export-onnx` produces

`models\plantvillage_trained\model.onnx` + `class_names.json`, deliberately built
with the same interface as the shipped model, so `predict.py`, `app.py` and
`evaluate_model.py` need no change:

| Property | Value |
|---|---|
| Input | `input`, NCHW float32, RGB, ImageNet-normalised (same as `predict.py`) |
| Output | `logits`, raw scores, no softmax |
| Labels | sorted class-folder names, identical ordering to `class_names.json` |

`predict.py` reads the resolution from the graph, so models trained at 128 or 160
pixels work without edits too. Example of a 15-class model trained here at
MobileNetV3-Small `alpha=0.75`, 128x128:

```powershell
.venv\Scripts\python inspect_onnx.py models\plantvillage_trained\model.onnx
$env:PLANT_MODEL_DIR = "models\plantvillage_trained"
.venv\Scripts\python predict.py --image path\to\leaf.jpg
```


## Troubleshooting

- `model not found ...` -> run `download_pretrained_model.py` first.
- `No module named pip` with the system Python 3.14 -> use `uv` (see Quick start)
  or run `python -m ensurepip --upgrade`.
- Uploaded images are written to `uploads/`, which is created on demand.

## Licence

Educational project. The pretrained weights come from the Hugging Face repository
`imaflower/plantvillage-mobilenetv3` (MIT licence). Please cite the PlantVillage
dataset and that repository when reusing the model.

