"""Check whether ONNX Runtime int8 quantization is reliable in this environment.

Builds two tiny classifiers (one with regular convolutions, one with a depthwise
convolution like every MobileNet), quantizes both with dynamic and QDQ int8, and
compares the outputs with the float32 original. Run it when an int8 build of a
real model misbehaves, to tell an environment problem apart from a model problem:

    python check_quantization_support.py

A healthy environment prints 20/20 argmax agreement for all four rows. Measured
here, the environment is healthy while full int8 builds of the PlantVillage
MobileNetV3 models still collapse to 5-19% accuracy, which is why
quantize_model.py rejects those builds instead of shipping them.
"""
import os
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_dynamic, quantize_static)
import onnxruntime as ort

rng = np.random.default_rng(0)


def build_model(path):
    w1 = rng.normal(0, 0.2, (8, 3, 3, 3)).astype(np.float32)
    b1 = rng.normal(0, 0.1, (8,)).astype(np.float32)
    w2 = rng.normal(0, 0.2, (8, 8)).astype(np.float32)  # 8 classes
    b2 = rng.normal(0, 0.1, (8,)).astype(np.float32)
    nodes = [
        helper.make_node("Conv", ["input", "w1", "b1"], ["c1"], pads=[1, 1, 1, 1], name="c1"),
        helper.make_node("Relu", ["c1"], ["r1"], name="r1"),
        helper.make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2, 2], strides=[2, 2], name="p1"),
        helper.make_node("GlobalAveragePool", ["p1"], ["gap"], name="gap"),
        helper.make_node("Flatten", ["gap"], ["flat"], name="flat"),
        helper.make_node("Gemm", ["flat", "w2", "b2"], ["logits"], name="gemm", transB=1),
    ]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 8])],
        [numpy_helper.from_array(w1, "w1"), numpy_helper.from_array(b1, "b1"),
         numpy_helper.from_array(w2, "w2"), numpy_helper.from_array(b2, "b2")])
    graph.value_info.extend([
        helper.make_tensor_value_info("c1", TensorProto.FLOAT, [1, 8, 16, 16]),
        helper.make_tensor_value_info("r1", TensorProto.FLOAT, [1, 8, 16, 16]),
        helper.make_tensor_value_info("p1", TensorProto.FLOAT, [1, 8, 8, 8]),
        helper.make_tensor_value_info("gap", TensorProto.FLOAT, [1, 8, 1, 1]),
        helper.make_tensor_value_info("flat", TensorProto.FLOAT, [1, 8]),
    ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


class RandomReader(CalibrationDataReader):
    def __init__(self, count=20):
        self.batches = iter([{"input": rng.normal(0, 1, (1, 3, 16, 16)).astype(np.float32)}
                             for _ in range(count)])

    def get_next(self):
        return next(self.batches, None)


def compare(path, inputs):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return [session.run(None, {"input": x})[0][0] for x in inputs]


build_model("_sanity_fp32.onnx")
quantize_dynamic("_sanity_fp32.onnx", "_sanity_dyn.onnx", weight_type=QuantType.QUInt8)
quantize_static("_sanity_fp32.onnx", "_sanity_static.onnx", RandomReader(),
                quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8, per_channel=True)

def build_depthwise_model(path):
    """Same shape of network but with a depthwise conv, like every MobileNet."""
    w_dw = rng.normal(0, 0.3, (3, 1, 3, 3)).astype(np.float32)   # depthwise
    b_dw = rng.normal(0, 0.1, (3,)).astype(np.float32)
    w_pw = rng.normal(0, 0.2, (8, 3, 1, 1)).astype(np.float32)   # pointwise
    b_pw = rng.normal(0, 0.1, (8,)).astype(np.float32)
    w2 = rng.normal(0, 0.2, (8, 8)).astype(np.float32)
    b2 = rng.normal(0, 0.1, (8,)).astype(np.float32)
    nodes = [
        helper.make_node("Conv", ["input", "w_dw", "b_dw"], ["dw"], pads=[1, 1, 1, 1],
                         group=3, name="dw"),
        helper.make_node("Relu", ["dw"], ["rdw"], name="rdw"),
        helper.make_node("Conv", ["rdw", "w_pw", "b_pw"], ["pw"], name="pw"),
        helper.make_node("Relu", ["pw"], ["rpw"], name="rpw"),
        helper.make_node("GlobalAveragePool", ["rpw"], ["gap"], name="gap2"),
        helper.make_node("Flatten", ["gap"], ["flat"], name="flat2"),
        helper.make_node("Gemm", ["flat", "w2", "b2"], ["logits"], name="gemm2", transB=1),
    ]
    graph = helper.make_graph(
        nodes, "depthwise",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 8])],
        [numpy_helper.from_array(w_dw, "w_dw"), numpy_helper.from_array(b_dw, "b_dw"),
         numpy_helper.from_array(w_pw, "w_pw"), numpy_helper.from_array(b_pw, "b_pw"),
         numpy_helper.from_array(w2, "w2"), numpy_helper.from_array(b2, "b2")])
    graph.value_info.extend([
        helper.make_tensor_value_info("dw", TensorProto.FLOAT, [1, 3, 16, 16]),
        helper.make_tensor_value_info("rdw", TensorProto.FLOAT, [1, 3, 16, 16]),
        helper.make_tensor_value_info("pw", TensorProto.FLOAT, [1, 8, 16, 16]),
        helper.make_tensor_value_info("rpw", TensorProto.FLOAT, [1, 8, 16, 16]),
        helper.make_tensor_value_info("gap", TensorProto.FLOAT, [1, 8, 1, 1]),
        helper.make_tensor_value_info("flat", TensorProto.FLOAT, [1, 8]),
    ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


inputs = [rng.normal(0, 1, (1, 3, 16, 16)).astype(np.float32) for _ in range(20)]
with tempfile.TemporaryDirectory() as work_dir:
    for tag, builder in (("regular-conv", build_model),
                         ("depthwise-conv", build_depthwise_model)):
        fp32_path = os.path.join(work_dir, "{}-fp32.onnx".format(tag))
        dyn_path = os.path.join(work_dir, "{}-dynamic.onnx".format(tag))
        qdq_path = os.path.join(work_dir, "{}-qdq.onnx".format(tag))
        builder(fp32_path)
        quantize_dynamic(fp32_path, dyn_path, weight_type=QuantType.QUInt8)
        quantize_static(fp32_path, qdq_path, RandomReader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        per_channel=True)
        print("== {} ==".format(tag))
        base = compare(fp32_path, inputs)
        for mode, path in (("dynamic", dyn_path), ("static-qdq", qdq_path)):
            other = compare(path, inputs)
            diff = max(float(np.max(np.abs(a - b))) for a, b in zip(base, other))
            same = sum(int(np.argmax(a) == np.argmax(b)) for a, b in zip(base, other))
            print("[RESULT] {:<10} max abs diff {:.4f}  argmax agreement {}/{}".format(
                mode, diff, same, len(inputs)))


