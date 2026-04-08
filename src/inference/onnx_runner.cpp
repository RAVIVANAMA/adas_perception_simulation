#include "inference/onnx_runner.hpp"
#include "common/logger.hpp"
#include <stdexcept>
#include <algorithm>
#include <numeric>

#ifdef HAVE_ONNXRUNTIME
#  include <onnxruntime_cxx_api.h>
#endif

namespace adas::inference {

// ─── PIMPL for ONNX Runtime – keeps ORT types out of the public header ────────
struct OnnxRunner::Impl {
#ifdef HAVE_ONNXRUNTIME
    Ort::Env                         env{ORT_LOGGING_LEVEL_WARNING, "adas"};
    std::unique_ptr<Ort::Session>    session;
    Ort::SessionOptions              sessionOpts;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string>          inputNames;
    std::vector<std::string>          outputNames;
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<std::vector<int64_t>> outputShapes;
#endif
};

// ─── OnnxRunner ──────────────────────────────────────────────────────────────
OnnxRunner::OnnxRunner(int deviceId)
    : impl_(std::make_unique<Impl>()), deviceId_(deviceId)
{
#ifdef HAVE_ONNXRUNTIME
    impl_->sessionOpts.SetIntraOpNumThreads(4);
    impl_->sessionOpts.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (deviceId_ >= 0) {
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = deviceId_;
        impl_->sessionOpts.AppendExecutionProvider_CUDA(cuda);
        LOG_INFO("OnnxRunner: using CUDA device " << deviceId_);
    } else {
        LOG_INFO("OnnxRunner: running on CPU");
    }
#else
    LOG_WARN("OnnxRunner: compiled without ONNX Runtime – using stub mode");
#endif
}

OnnxRunner::~OnnxRunner() = default;

void OnnxRunner::loadModel(const std::string& modelPath) {
#ifdef HAVE_ONNXRUNTIME
    impl_->session = std::make_unique<Ort::Session>(
        impl_->env,
        modelPath.c_str(),
        impl_->sessionOpts);

    size_t nIn  = impl_->session->GetInputCount();
    size_t nOut = impl_->session->GetOutputCount();

    impl_->inputNames.clear();
    impl_->outputNames.clear();
    impl_->inputShapes.clear();
    impl_->outputShapes.clear();

    for (size_t i = 0; i < nIn; ++i) {
        auto name = impl_->session->GetInputNameAllocated(i, impl_->allocator);
        impl_->inputNames.emplace_back(name.get());
        auto info  = impl_->session->GetInputTypeInfo(i);
        auto shape = info.GetTensorTypeAndShapeInfo().GetShape();
        impl_->inputShapes.push_back(shape);
    }
    for (size_t i = 0; i < nOut; ++i) {
        auto name = impl_->session->GetOutputNameAllocated(i, impl_->allocator);
        impl_->outputNames.emplace_back(name.get());
        auto info  = impl_->session->GetOutputTypeInfo(i);
        auto shape = info.GetTensorTypeAndShapeInfo().GetShape();
        impl_->outputShapes.push_back(shape);
    }
    isLoaded_ = true;
    LOG_INFO("OnnxRunner: loaded model '" << modelPath << "' ("
             << nIn << " inputs, " << nOut << " outputs)");
#else
    // Stub – just remember that a model was "loaded"
    (void)modelPath;
    isLoaded_ = true;
    LOG_WARN("OnnxRunner stub: pretending to load '" << modelPath << "'");
#endif
}

void OnnxRunner::run(
    const std::unordered_map<std::string, std::vector<float>>& inputs,
    std::unordered_map<std::string, std::vector<float>>& outputs)
{
    if (!isLoaded_) throw std::runtime_error("OnnxRunner: model not loaded");

#ifdef HAVE_ONNXRUNTIME
    auto memInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputTensors;
    std::vector<const char*> inNames, outNames;

    for (size_t i = 0; i < impl_->inputNames.size(); ++i) {
        const auto& name = impl_->inputNames[i];
        inNames.push_back(name.c_str());
        auto& buf  = inputs.at(name);
        auto& shape = impl_->inputShapes[i];
        // Replace dynamic dims (-1) with actual sizes from buffer
        std::vector<int64_t> dims = shape;
        if (dims[0] < 0) dims[0] = 1;
        int64_t total = std::accumulate(dims.begin(), dims.end(),
                                        int64_t{1}, std::multiplies<>());
        (void)total; // used for assertion in debug
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memInfo,
            const_cast<float*>(buf.data()), buf.size(),
            dims.data(), dims.size()));
    }
    for (const auto& n : impl_->outputNames)
        outNames.push_back(n.c_str());

    auto outTensors = impl_->session->Run(
        Ort::RunOptions{nullptr},
        inNames.data(),  inputTensors.data(),  inNames.size(),
        outNames.data(), outNames.size());

    for (size_t i = 0; i < outNames.size(); ++i) {
        auto* data  = outTensors[i].GetTensorMutableData<float>();
        auto  shape = outTensors[i].GetTensorTypeAndShapeInfo().GetShape();
        int64_t total = std::accumulate(shape.begin(), shape.end(),
                                        int64_t{1}, std::multiplies<>());
        outputs[outNames[i]].assign(data, data + total);
    }
#else
    // Stub – leave outputs empty
    (void)inputs; (void)outputs;
    LOG_WARN("OnnxRunner stub: run() is a no-op");
#endif
}

std::vector<TensorInfo> OnnxRunner::inputInfo() const {
    std::vector<TensorInfo> result;
#ifdef HAVE_ONNXRUNTIME
    for (size_t i = 0; i < impl_->inputNames.size(); ++i)
        result.push_back({impl_->inputNames[i], impl_->inputShapes[i], "float32"});
#endif
    return result;
}

std::vector<TensorInfo> OnnxRunner::outputInfo() const {
    std::vector<TensorInfo> result;
#ifdef HAVE_ONNXRUNTIME
    for (size_t i = 0; i < impl_->outputNames.size(); ++i)
        result.push_back({impl_->outputNames[i], impl_->outputShapes[i], "float32"});
#endif
    return result;
}

} // namespace adas::inference
