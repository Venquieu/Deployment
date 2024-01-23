import os
import argparse

import tensorrt as trt
from cuda import cudart

from core.calibrator import MyCalibrator


def make_parser():
    parser = argparse.ArgumentParser("Export TensorRT engine from Onnx")
    parser.add_argument("--onnx", required=True, help="onnx model path")
    parser.add_argument("-output", required=True, help="output TensorRT engine path")
    parser.add_argument(
        "--shape", required=True, help="model input shape with format `C,H,W`"
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Whether to enable fp16 mode"
    )
    parser.add_argument(
        "--int8", action="store_true", help="Whether to enable int8 mode"
    )
    parser.add_argument(
        "--calibration_data", default=None, help="Calibration data path for int8"
    )
    parser.add_argument(
        "--calibration_cache", default=None, help="Calibration cache path for int8"
    )
    return parser


def export(
    onnx_path: str,
    output_path: str,
    input_shape: tuple,
    precision_mode="fp16",
    calibrator=None,
):
    """
    Args:
        onnx_path: onnx path
        input_shape: preferred input size of model with format (C, H, W)
    """
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()

    if precision_mode == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision_mode == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
    else:
        raise NotImplementedError(f"precision mode {precision_mode} not implemented.")

    if not os.path.exists(onnx_path):
        print(f"ONNX file {onnx_path} not exists.")
        return False

    print("loading ONNX file...")
    parser = trt.OnnxParser(network, logger)
    # parse_valid = parser.parse_from_file(onnx_path)
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
        print("Succeeded parsing .onnx file!")

    input_tensor = network.get_input(0)
    profile.set_shape(
        input_tensor.name,
        [1, *input_shape],
        [4, *input_shape],
        [8, *input_shape],
    )
    config.add_optimization_profile(profile)

    engine_string = builder.build_serialized_network(network, config)
    if engine_string == None:
        print("Failed building engine!")
        return False

    print("Succeeded building engine!")
    with open(output_path, "wb") as f:
        f.write(engine_string)

    return True


if __name__ == "__main__":
    args = make_parser().parse_args()
    input_shape = [int(x) for x in args.shape.split(",")]

    if args.fp16:
        precision_mode = "fp16"
        calibrator = None
    elif args.int8:
        assert (
            args.calibration_data is not None and args.calibration_cache is not None
        ), "calibration data path and cache path must be provided under int8 mode"
        precision_mode = "int8"
        calibrator = MyCalibrator(
            args.calibration_data, 1, (1, *input_shape), args.calibration_cache
        )
    else:
        print("You must enable fp16 or int8!")
        exit(1)

    is_success = export(
        args.onnx,
        args.ouput,
        input_shape,
        precision_mode=precision_mode,
        calibrator=calibrator,
    )

    if is_success:
        print(f"TensorRT engine saved as {args.ouput}")
