#pragma once

#include "common/types.hpp"
#include "inference/onnx_runner.hpp"
#include <memory>
#include <vector>
#include <functional>

namespace adas::perception {

// ─── Object detector configuration ─────────────────────────────────────────
struct ObjectDetectorConfig {
    std::string   modelPath;
    float         confidenceThreshold{0.45f};
    float         nmsIouThreshold{0.45f};
    int           inputWidth{640};
    int           inputHeight{640};
    int           deviceId{-1};              // -1 = CPU
    inference::Backend backend{inference::Backend::ONNX};
};

// ─── Object detector ────────────────────────────────────────────────────────
/// Wraps a YOLO-style (or similar) DNN for 2-D object detection.
/// Input:  raw camera frame (BGR uint8).
/// Output: vector of DetectedObject sorted by descending confidence.
class ObjectDetector {
public:
    using DetectionCallback = std::function<void(const std::vector<DetectedObject>&)>;

    explicit ObjectDetector(ObjectDetectorConfig cfg);
    ~ObjectDetector();

    // Detect objects in one camera frame
    std::vector<DetectedObject> detect(const CameraFrame& frame);

    // Register an optional callback fired after every detect() call
    void setCallback(DetectionCallback cb) { callback_ = std::move(cb); }

    bool isReady() const;
    const ObjectDetectorConfig& config() const { return cfg_; }

private:
    // Pre/post-processing helpers
    std::vector<float> preprocess(const CameraFrame& frame) const;
    std::vector<DetectedObject> postprocess(
        const std::unordered_map<std::string, std::vector<float>>& raw,
        int origW, int origH) const;

    ObjectDetectorConfig                       cfg_;
    std::unique_ptr<inference::IInferenceRunner> runner_;
    DetectionCallback                            callback_;
    uint64_t                                     detectionCount_{0};
};

} // namespace adas::perception
