#ifndef ONNX_TO_TRT_CPP_UTILS_H
#define ONNX_TO_TRT_CPP_UTILS_H

#include <iostream>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#define CUDA_CHECK(status) check(status, __LINE__, __FILE__)

inline bool check(const cudaError_t status,const int line, const char* file) {
    if (status!= cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at line " << line << " in file " << file << std::endl;
        return false;
    }
    return true;
}

namespace onnx2trt {

class Logger : public nvinfer1::ILogger {
  public:
    Logger(const nvinfer1::ILogger::Severity severity);
    void log(const nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
    nvinfer1::ILogger::Severity getSeverity() { return severity_; }

  private:
    nvinfer1::ILogger::Severity severity_;
};

} // namespace onnx2trt

#endif
