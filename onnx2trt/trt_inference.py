import argparse

import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart

cudart.cudaDeviceSynchronize()


def make_parser():
    parser = argparse.ArgumentParser("TensorRT engine inference")
    parser.add_argument("--engine", required=True, help="path of TensorRT engine")
    parser.add_argument("--img", required=True, help="image path for inference")
    parser.add_argument(
        "--shape", required=True, help="model input shape with format `C,H,W`"
    )
    return parser


class NetRuntime(object):
    def __init__(self, engine_path, input_shape) -> None:
        logger = trt.Logger(trt.Logger.VERBOSE)

        with open(engine_path, "rb") as f:
            engine_string = f.read()
        engine = trt.Runtime(logger).deserialize_cuda_engine(engine_string)

        num_io = engine.num_io_tensors
        tensor_names = [engine.get_tensor_name(i) for i in range(num_io)]
        num_input = [
            engine.get_tensor_mode(tensor_names[i]) for i in range(num_io)
        ].count(trt.TensorIOMode.INPUT)

        context = engine.create_execution_context()
        context.set_input_shape(tensor_names[0], [1, *input_shape])

        self.num_input = num_input
        self.num_io = num_io
        self.tensor_names = tensor_names
        self.context = context
        self.engine = engine

        self.alloc_memory(input_shape)

        print("I/O tensor info:")
        self.print_io_info()

    def alloc_memory(self, input_shape):
        # alloc host memory
        self.buffer_host = []
        self.buffer_host.append(np.empty((1, *input_shape), dtype=np.float32))
        for i in range(self.num_input, self.num_io):
            out_shape = self.context.get_tensor_shape(self.tensor_names[i])
            out_dtype = self.engine.get_tensor_dtype(self.tensor_names[i])
            self.buffer_host.append(np.empty(out_shape, dtype=trt.nptype(out_dtype)))

        # alloc device memory
        self.buffer_device = []
        for i in range(self.num_io):
            self.buffer_device.append(cudart.cudaMalloc(self.buffer_host[i].nbytes)[1])

        # binding tensor and memory
        for i in range(self.num_io):
            self.context.set_tensor_address(self.tensor_names[i], int(self.buffer_device[i]))

    def execute(self, img):
        for i in range(self.num_input):
            self.buffer_host[i] = img
            cudart.cudaMemcpy(
                self.buffer_device[i],
                self.buffer_host[i].ctypes.data,
                self.buffer_host[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )

        self.context.execute_async_v3(0)

        for i in range(self.num_input, self.num_io):
            cudart.cudaMemcpy(
                self.buffer_host[i].ctypes.data,
                self.buffer_device[i],
                self.buffer_host[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )

    def print_io_info(self):
        for i in range(self.num_io):
            print(
                "[%2d]%s->" % (i, "Input " if i < self.num_input else "Output"),
                self.engine.get_tensor_dtype(self.tensor_names[i]),
                self.engine.get_tensor_shape(self.tensor_names[i]),
                self.context.get_tensor_shape(self.tensor_names[i]),
                self.tensor_names[i],
            )

    def __del__(self):
        for b in self.buffer_device:
            cudart.cudaFree(b)


if __name__ == "__main__":
    args = make_parser().parse_args()
    input_shape = [int(x) for x in args.shape.split(",")]

    net = NetRuntime(args.engine, input_shape)
    data = (
        cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
        .astype(np.float32)
        .reshape(1, *input_shape)
    )

    net.execute(data)
    result = net.buffer_device[net.num_input:]
    print("Succeeded running model in TensorRT!")
