#include "inference/onnx_runner.hpp"
#include "common/logger.hpp"

#ifdef HAVE_TENSORRT
#  include <NvInfer.h>
#  include <NvOnnxParser.h>
#  include <cuda_runtime_api.h>
#  include <fstream>
#  include <stdexcept>
#endif

namespace adas::inference {

#ifdef HAVE_TENSORRT

// ─── TRT logging bridge ───────────────────────────────────────────────────────
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
        case Severity::kERROR:   LOG_ERROR("TensorRT: " << msg); break;
        case Severity::kWARNING: LOG_WARN ("TensorRT: " << msg); break;
        case Severity::kINFO:    LOG_DEBUG("TensorRT: " << msg); break;
        default: break;
        }
    }
};

struct TensorRTRunner::Impl {
    TrtLogger                              trtLogger;
    nvinfer1::IRuntime*                    runtime{nullptr};
    nvinfer1::ICudaEngine*                 engine{nullptr};
    nvinfer1::IExecutionContext*           context{nullptr};
    std::vector<void*>                     deviceBuffers;
    std::vector<std::vector<int64_t>>      bindingShapes;
    std::vector<std::string>               bindingNames;
    cudaStream_t                           stream{nullptr};
};

TensorRTRunner::TensorRTRunner(int deviceId, bool useINT8, bool useFP16)
    : impl_(std::make_unique<Impl>())
{
    cudaSetDevice(deviceId);
    cudaStreamCreate(&impl_->stream);
    LOG_INFO("TensorRTRunner: device=" << deviceId
             << " FP16=" << useFP16 << " INT8=" << useINT8);
}

TensorRTRunner::~TensorRTRunner() {
    for (auto buf : impl_->deviceBuffers) cudaFree(buf);
    if (impl_->context) impl_->context->destroy();
    if (impl_->engine)  impl_->engine->destroy();
    if (impl_->runtime) impl_->runtime->destroy();
    if (impl_->stream)  cudaStreamDestroy(impl_->stream);
}

void TensorRTRunner::loadModel(const std::string& modelPath) {
    impl_->runtime = nvinfer1::createInferRuntime(impl_->trtLogger);

    // Try loading a serialised engine first
    std::ifstream engineFile(modelPath, std::ios::binary);
    if (engineFile.good()) {
        engineFile.seekg(0, std::ios::end);
        size_t fsize = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);
        std::vector<char> engineData(fsize);
        engineFile.read(engineData.data(), fsize);

        impl_->engine = impl_->runtime->deserializeCudaEngine(
            engineData.data(), fsize);
        LOG_INFO("TensorRTRunner: loaded serialised engine '" << modelPath << "'");
    } else {
        throw std::runtime_error("TensorRTRunner: engine file not found: " + modelPath);
    }

    impl_->context = impl_->engine->createExecutionContext();
    isLoaded_ = true;

    int nBindings = impl_->engine->getNbBindings();
    impl_->deviceBuffers.resize(nBindings, nullptr);
    for (int i = 0; i < nBindings; ++i) {
        impl_->bindingNames.push_back(impl_->engine->getBindingName(i));
        auto dims = impl_->engine->getBindingDimensions(i);
        std::vector<int64_t> shape(dims.d, dims.d + dims.nbDims);
        impl_->bindingShapes.push_back(shape);
        size_t sz = 1;
        for (auto d : shape) sz *= (d > 0 ? d : 1);
        cudaMalloc(&impl_->deviceBuffers[i], sz * sizeof(float));
    }
}

void TensorRTRunner::run(
    const std::unordered_map<std::string, std::vector<float>>& inputs,
    std::unordered_map<std::string, std::vector<float>>& outputs)
{
    if (!isLoaded_) throw std::runtime_error("TensorRTRunner: model not loaded");

    for (size_t i = 0; i < impl_->bindingNames.size(); ++i) {
        const auto& name = impl_->bindingNames[i];
        if (inputs.count(name)) {
            const auto& buf = inputs.at(name);
            cudaMemcpyAsync(impl_->deviceBuffers[i], buf.data(),
                            buf.size() * sizeof(float),
                            cudaMemcpyHostToDevice, impl_->stream);
        }
    }

    impl_->context->enqueueV2(impl_->deviceBuffers.data(), impl_->stream, nullptr);

    for (size_t i = 0; i < impl_->bindingNames.size(); ++i) {
        if (impl_->engine->bindingIsInput(i)) continue;
        const auto& name = impl_->bindingNames[i];
        const auto& shape = impl_->bindingShapes[i];
        size_t sz = 1;
        for (auto d : shape) sz *= (d > 0 ? d : 1);
        outputs[name].resize(sz);
        cudaMemcpyAsync(outputs[name].data(), impl_->deviceBuffers[i],
                        sz * sizeof(float),
                        cudaMemcpyDeviceToHost, impl_->stream);
    }
    cudaStreamSynchronize(impl_->stream);
}

std::vector<TensorInfo> TensorRTRunner::inputInfo() const {
    std::vector<TensorInfo> r;
    for (size_t i = 0; i < impl_->bindingNames.size(); ++i) {
        if (impl_->engine->bindingIsInput(static_cast<int>(i)))
            r.push_back({impl_->bindingNames[i], impl_->bindingShapes[i], "float32"});
    }
    return r;
}

std::vector<TensorInfo> TensorRTRunner::outputInfo() const {
    std::vector<TensorInfo> r;
    for (size_t i = 0; i < impl_->bindingNames.size(); ++i) {
        if (!impl_->engine->bindingIsInput(static_cast<int>(i)))
            r.push_back({impl_->bindingNames[i], impl_->bindingShapes[i], "float32"});
    }
    return r;
}

void TensorRTRunner::serializeEngine(const std::string& enginePath) const {
    auto* serialised = impl_->engine->serialize();
    std::ofstream out(enginePath, std::ios::binary);
    out.write(static_cast<const char*>(serialised->data()), serialised->size());
    serialised->destroy();
    LOG_INFO("TensorRTRunner: engine serialised to '" << enginePath << "'");
}

#endif // HAVE_TENSORRT

} // namespace adas::inference
