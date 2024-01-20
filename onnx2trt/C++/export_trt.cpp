#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "utils.h"

const int nHeight = 28;
const int nWidth = 28;
const std::string onnx_path = "../model.onnx";
const std::string trtFile = "./model.plan";
const std::string dataFile = "./data.npz";
static onnx2trt::Logger gLogger(nvinfer1::ILogger::Severity::kERROR);

// for FP16 mode
const bool bFP16Mode =true;
// for INT8 mode
const bool bINT8Mode = false;
const int nCalibration = 1;
const std::string cacheFile = "./int8.cache";
const std::string calibrationDataFile = std::string("./data.npz");

int main()
{
    CUDA_CHECK(cudaSetDevice(0));
    nvinfer1::ICudaEngine *engine = nullptr;

    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(
        1U << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)
    );
    nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::IInt8Calibrator *pCalibrator = nullptr;

    if (bFP16Mode)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // if (bINT8Mode)
    // {
    //     config->setFlag(nvinfer1::BuilderFlag::kINT8);
    //     nvinfer1::Dims inputShape {4, {1, 1, nHeight, nWidth}};
    //     pCalibrator = new MyCalibrator(calibrationDataFile, nCalibration, inputShape, cacheFile);
    //     if (pCalibrator == nullptr)
    //     {
    //         std::cout << std::string("Failed getting Calibrator for Int8!") << std::endl;
    //         return 1;
    //     }
    //     config->setInt8Calibrator(pCalibrator);
    // }

    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser->parseFromFile(onnx_path.c_str(), int(gLogger.getSeverity())))
    {
        std::cout << std::string("Failed parsing .onnx file!") << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i)
        {
            auto *error = parser->getError(i);
            std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
        }
        return 1;
    }
    std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

    nvinfer1::ITensor *inputTensor = network->getInput(0);
    profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims {4, {1, 1, nHeight, nWidth}});
    profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims {4, {4, 1, nHeight, nWidth}});
    profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims {4, {8, 1, nHeight, nWidth}});
    config->addOptimizationProfile(profile);

    network->unmarkOutput(*network->getOutput(0)); // remove output tensor "y"

    nvinfer1::IHostMemory *engineString = nullptr;
#if TENSORRT_VERSION == 8
    engineString = builder->buildSerializedNetwork(*network, *config);
#else
    engine = builder->buildCudaEngine(*network);
    engineString = engine->serialize();
#endif

    if (engineString == nullptr || engineString->size() == 0)
    {
        std::cout << "Failed building serialized engine!" << std::endl;
        return 1;
    }
    std::cout << "Succeeded building serialized engine!" << std::endl;

    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
    if (engine == nullptr)
    {
        std::cout << "Failed building engine!" << std::endl;
        return 1;
    }
    std::cout << "Succeeded building engine!" << std::endl;

    if (bINT8Mode && pCalibrator != nullptr)
    {
        delete pCalibrator;
    }

    // save engine to file
    std::ofstream engineFile(trtFile, std::ios::binary);
    if (!engineFile)
    {
        std::cout << "Failed opening file to write" << std::endl;
        return 1;
    }
    engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
    if (engineFile.fail())
    {
        std::cout << "Failed saving .plan file!" << std::endl;
        return 1;
    }
    std::cout << "Succeeded saving .plan file!" << std::endl;

    return 0;
}
