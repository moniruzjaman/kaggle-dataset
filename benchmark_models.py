"""Interleaved size + latency benchmark for every model in models/.

Reports median single-image latency so the numbers are comparable across builds,
instead of the noisy sequential measurements.
"""
import glob
import os
import random
import statistics
import time

import onnxruntime as ort

from predict import preprocess_image

random.seed(0)
image_path = random.sample(
    glob.glob(os.path.join("PlantVillage", "PlantVillage", "*", "*.JPG")), 1)[0]

candidates = []
for pattern in ("models/*/model.onnx", "models/*.onnx"):
    for path in sorted(glob.glob(pattern)):
        size = os.path.getsize(path)
        size += os.path.getsize(path + ".data") if os.path.exists(path + ".data") else 0
        candidates.append((path, size))

sessions = []
for path, size in candidates:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    img_size = shape[2] if isinstance(shape[2], int) else 224
    batch = preprocess_image(image_path, img_size)
    name = session.get_inputs()[0].name
    for _ in range(10):
        session.run(None, {name: batch})
    timings = []
    sessions.append([name, session, batch, path, size, timings])

for _ in range(80):
    for entry in sessions:
        _name, session, batch, _path, _size, timings = entry
        start = time.perf_counter()
        session.run(None, {entry[0]: batch})
        timings.append((time.perf_counter() - start) * 1000.0)

print("{:<52} {:>9} {:>10}".format("model", "size", "median ms"))
for name, _session, _batch, path, size, timings in sorted(sessions, key=lambda e: e[4]):
    print("{:<52} {:>6.2f} MB {:>8.2f}".format(
        path, size / (1024.0 * 1024.0), statistics.median(timings)))
