import argparse

import torch

from core.model import Net

def make_parser():
    parser = argparse.ArgumentParser("Export TensorRT engine from Onnx")
    parser.add_argument("-ckpt", required=True, help="pyTorch model checkpoint path")
    parser.add_argument("-output", required=True, help="export onnx file save path")
    parser.add_argument(
        "--shape", required=True, help="model input shape with format `C,H,W`"
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    input_shape = [int(x) for x in args.shape.split(",")]

    model = Net()
    torch.onnx.export(
        model,
        torch.randn(1, *input_shape, device="cuda"),
        args.output,
        input_names=["x"],
        output_names=["y"],
        do_constant_folding=True,
        verbose=True,
        keep_initializers_as_inputs=True,
        opset_version=12,
        dynamic_axes={"x": {0: "nBatchSize"}, "y": {0: "nBatchSize"}},
    )
    print("Succeeded converting model into ONNX!")
