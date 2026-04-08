#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

// Forward-declare ONNX Runtime types so we don't pull in onnxruntime_cxx_api.h
// in every translation unit that includes this header.
#ifdef HAVE_ONNXRUNTIME
  namespace Ort { class Session; class Env; class SessionOptions; }
#endif

namespace adas::inference {

// ─── Tensor descriptor ────────────────────────────────────────────────────────
struct TensorInfo {
    std::string              name;
    std::vector<int64_t>     shape;   // -1 means dynamic dimension
    std::string              dtype;   // "float32", "int64", etc.
};

// ─── Generic inference backend interface ─────────────────────────────────────
class IInferenceRunner {
public:
    virtual ~IInferenceRunner() = default;

    /// Load (or reload) a model from disk.
    virtual void loadModel(const std::string& modelPath) = 0;

    /// Run one inference.
    /// @param inputs  map from input-name  → flat float buffer
    /// @param outputs map from output-name → flat float buffer (pre-sized by caller)
    virtual void run(const std::unordered_map<std::string, std::vector<float>>& inputs,
                     std::unordered_map<std::string, std::vector<float>>& outputs) = 0;

    virtual std::vector<TensorInfo> inputInfo()  const = 0;
    virtual std::vector<TensorInfo> outputInfo() const = 0;
    virtual bool                    isLoaded()   const = 0;
};

// ─── ONNX Runtime backend ─────────────────────────────────────────────────────
class OnnxRunner final : public IInferenceRunner {
public:
    /// @param deviceId  GPU device index; -1 = CPU only
    explicit OnnxRunner(int deviceId = -1);
    ~OnnxRunner() override;

    void loadModel(const std::string& modelPath) override;
    void run(const std::unordered_map<std::string, std::vector<float>>& inputs,
             std::unordered_map<std::string, std::vector<float>>& outputs) override;

    std::vector<TensorInfo> inputInfo()  const override;
    std::vector<TensorInfo> outputInfo() const override;
    bool                    isLoaded()   const override { return isLoaded_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int  deviceId_{-1};
    bool isLoaded_{false};
};

// ─── TensorRT backend ─────────────────────────────────────────────────────────
#ifdef HAVE_TENSORRT
class TensorRTRunner final : public IInferenceRunner {
public:
    explicit TensorRTRunner(int deviceId = 0,
                            bool useINT8 = false,
                            bool useFP16 = true);
    ~TensorRTRunner() override;

    void loadModel(const std::string& modelPath) override;
    void run(const std::unordered_map<std::string, std::vector<float>>& inputs,
             std::unordered_map<std::string, std::vector<float>>& outputs) override;

    std::vector<TensorInfo> inputInfo()  const override;
    std::vector<TensorInfo> outputInfo() const override;
    bool                    isLoaded()   const override { return isLoaded_; }

    /// Serialize optimised engine to disk for future use.
    void serializeEngine(const std::string& enginePath) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool isLoaded_{false};
};
#endif

// ─── Factory helper ───────────────────────────────────────────────────────────
enum class Backend { ONNX, TensorRT };

inline std::unique_ptr<IInferenceRunner>
makeRunner(Backend backend, int deviceId = -1) {
    switch (backend) {
    case Backend::ONNX:
        return std::make_unique<OnnxRunner>(deviceId);
#ifdef HAVE_TENSORRT
    case Backend::TensorRT:
        return std::make_unique<TensorRTRunner>(deviceId);
#endif
    default:
        throw std::invalid_argument("Unsupported or disabled inference backend");
    }
}

} // namespace adas::inference
