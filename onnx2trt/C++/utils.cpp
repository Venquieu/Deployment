#include <iostream>
#include <sstream>

#include "utils.h"

namespace onnx2trt {

Logger::Logger(const nvinfer1::ILogger::Severity severity) : severity_(severity) {};

void Logger::log(const nvinfer1::ILogger::Severity severity, const char* msg) {
    if (severity >= severity_) {
        return;
    }

    std::stringstream ss;
    switch (severity)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        ss << "INTERNAL_ERROR: ";
        break;
    case nvinfer1::ILogger::Severity::kERROR:
        ss << "ERROR: ";
        break;
    case nvinfer1::ILogger::Severity::kWARNING:
        ss << "WARNING: ";
        break;
    case nvinfer1::ILogger::Severity::kINFO:
        ss << "INFO: ";
        break;
    default:
        ss << "VERBOSE: ";
        break;
    }
    std::cout << ss.str() << msg << std::endl;
}

} // namespace onnx2trt
