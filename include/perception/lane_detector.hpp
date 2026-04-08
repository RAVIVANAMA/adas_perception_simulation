#pragma once

#include "common/types.hpp"
#include "inference/onnx_runner.hpp"
#include <memory>
#include <vector>

namespace adas::perception {

struct LaneDetectorConfig {
    std::string modelPath;
    int   inputWidth{800};
    int   inputHeight{288};
    float confidenceThreshold{0.5f};
    int   deviceId{-1};
    inference::Backend backend{inference::Backend::ONNX};
    // Perspective transform parameters (calibration)
    float srcPoints[4][2]{{192,240},{624,240},{131,480},{709,480}};
    float dstPoints[4][2]{{200,0},{600,0},{200,480},{600,480}};
};

/// Lane detection and fitting.
/// Uses a segmentation DNN (e.g. SCNN / LaneATT / UFLDv2) to locate
/// left and right lane boundaries and fit polynomial curves to them.
class LaneDetector {
public:
    explicit LaneDetector(LaneDetectorConfig cfg);
    ~LaneDetector();

    /// Detect lanes in one camera frame.
    LaneInfo detect(const CameraFrame& frame);

    /// Compute lateral error and heading error relative to lane centre.
    /// Called internally by detect() but exposed for testing.
    static LaneInfo computeErrors(const LaneBoundary& left,
                                  const LaneBoundary& right,
                                  int imageWidth, int imageHeight);

    bool isReady() const;
    const LaneDetectorConfig& config() const { return cfg_; }

private:
    std::vector<float> preprocess(const CameraFrame& frame) const;
    LaneInfo           postprocess(
        const std::unordered_map<std::string, std::vector<float>>& raw,
        int origW, int origH) const;

    LaneDetectorConfig                             cfg_;
    std::unique_ptr<inference::IInferenceRunner>   runner_;
};

} // namespace adas::perception
