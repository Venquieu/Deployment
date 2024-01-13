import argparse

import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart

cudart.cudaDeviceSynchronize()


def make_parser():
    parser = argparse.ArgumentParser("TensorRT engine inference")
    parser.add_argument("--path", required=True, help="image path for inference")
    parser.add_argument(
        "--shape", required=True, help="model input shape with format `C,H,W`"
    )
    return parser


class NetRT(object):
    def __init__(self, engine_path, input_shape) -> None:
        logger = trt.Logger(trt.Logger.VERBOSE)

        with open(engine_path, "rb") as f:
            engineString = f.read()
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

        num_io = engine.num_io_tensors
        tensor_names = [engine.get_tensor_name(i) for i in range(num_io)]
        num_input = [
            engine.get_tensor_mode(tensor_names[i]) for i in range(num_io)
        ].count(trt.TensorIOMode.INPUT)

        context = engine.create_execution_context()
        context.set_input_shape(tensor_names[0], [1, *input_shape])

        for i in range(num_io):
            print(
                "[%2d]%s->" % (i, "Input " if i < num_input else "Output"),
                engine.get_tensor_dtype(tensor_names[i]),
                engine.get_tensor_shape(tensor_names[i]),
                context.get_tensor_shape(tensor_names[i]),
                tensor_names[i],
            )

        self.num_input = num_input
        self.num_io = num_io
        self.tensor_names = tensor_names
        self.context = context
        self.engine = engine

    def execute(self, img):
        buffer_host = []
        buffer_host.append(np.ascontiguousarray(img))
        for i in range(self.num_input, self.num_io):
            out_shape = self.context.get_tensor_shape(self.tensor_names[i])
            out_dtype = self.engine.get_tensor_dtype(self.tensor_names[i])
            buffer_host.append(np.empty(out_shape, dtype=trt.nptype(out_dtype)))

        buffer_device = []
        for i in range(self.num_io):
            buffer_device.append(cudart.cudaMalloc(buffer_host[i].nbytes)[1])

        for i in range(self.num_input):
            cudart.cudaMemcpy(
                buffer_device[i],
                buffer_host[i].ctypes.data,
                buffer_host[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )

        for i in range(self.num_io):
            self.context.set_tensor_address(self.tensor_names[i], int(buffer_device[i]))

        self.context.execute_async_v3(0)

        for i in range(self.num_input, self.num_io):
            cudart.cudaMemcpy(
                buffer_host[i].ctypes.data,
                buffer_device[i],
                buffer_host[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )

        for i in range(self.num_io):
            print(self.tensor_names[i])
            print(buffer_host[i])

        for b in buffer_device:
            cudart.cudaFree(b)

    def __del__(self):
        pass


if __name__ == "__main__":
    args = make_parser().parse_args()
    input_shape = [int(x) for x in args.shape.split(",")]

    data = (
        cv2.imread(args.path, cv2.IMREAD_GRAYSCALE)
        .astype(np.float32)
        .reshape(1, *input_shape)
    )

    print("Succeeded running model in TensorRT!")
