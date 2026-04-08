#include "perception/object_detector.hpp"
#include "common/logger.hpp"
#include "common/math_utils.hpp"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>

namespace adas::perception {

namespace {
// Class labels matching COCO-80 subset used by YOLO models
static constexpr std::array<ObjectClass, 80> kCocoLabelMap = [] {
    std::array<ObjectClass, 80> m{};
    m.fill(ObjectClass::Unknown);
    m[0]  = ObjectClass::Pedestrian;   // person
    m[1]  = ObjectClass::Cyclist;      // bicycle
    m[2]  = ObjectClass::Car;          // car
    m[3]  = ObjectClass::Motorcycle;   // motorcycle
    m[5]  = ObjectClass::Truck;        // bus
    m[7]  = ObjectClass::Truck;        // truck
    m[9]  = ObjectClass::TrafficLight; // traffic light
    return m;
}();

float sigmoid(float x) noexcept { return 1.f / (1.f + std::exp(-x)); }
} // anonymous namespace

// ─── Constructor ──────────────────────────────────────────────────────────────
ObjectDetector::ObjectDetector(ObjectDetectorConfig cfg)
    : cfg_(std::move(cfg))
{
    runner_ = inference::makeRunner(cfg_.backend, cfg_.deviceId);
    if (!cfg_.modelPath.empty()) {
        runner_->loadModel(cfg_.modelPath);
    }
    LOG_INFO("ObjectDetector: ready (model='"
             << cfg_.modelPath << "', device=" << cfg_.deviceId << ')');
}

ObjectDetector::~ObjectDetector() = default;

bool ObjectDetector::isReady() const {
    return runner_ && runner_->isLoaded();
}

// ─── Public API ───────────────────────────────────────────────────────────────
std::vector<DetectedObject> ObjectDetector::detect(const CameraFrame& frame)
{
    if (!isReady()) {
        LOG_WARN("ObjectDetector::detect called but model is not loaded");
        return {};
    }

    auto input = preprocess(frame);

    std::unordered_map<std::string, std::vector<float>> ins, outs;
    ins["images"] = std::move(input);
    runner_->run(ins, outs);

    auto detections = postprocess(outs, frame.width, frame.height);

    // Sort by descending confidence
    std::sort(detections.begin(), detections.end(),
              [](const DetectedObject& a, const DetectedObject& b){
                  return a.confidence > b.confidence;
              });

    ++detectionCount_;
    LOG_DEBUG("ObjectDetector: frame " << detectionCount_
              << " -> " << detections.size() << " detections");

    if (callback_) callback_(detections);
    return detections;
}

// ─── Pre-processing: resize + normalise to [0,1], HWC -> CHW ──────────────────
std::vector<float> ObjectDetector::preprocess(const CameraFrame& frame) const
{
    int H = cfg_.inputHeight, W = cfg_.inputWidth;
    std::vector<float> out(3 * H * W, 0.f);

    // Letterbox scale factors
    float scaleX = static_cast<float>(W) / frame.width;
    float scaleY = static_cast<float>(H) / frame.height;
    float scale  = std::min(scaleX, scaleY);
    int newW = static_cast<int>(frame.width  * scale);
    int newH = static_cast<int>(frame.height * scale);
    int padX = (W - newW) / 2;
    int padY = (H - newH) / 2;

    for (int y = 0; y < newH; ++y) {
        for (int x = 0; x < newW; ++x) {
            int srcX = static_cast<int>(x / scale);
            int srcY = static_cast<int>(y / scale);
            srcX = std::min(srcX, frame.width  - 1);
            srcY = std::min(srcY, frame.height - 1);

            int srcIdx = (srcY * frame.width + srcX) * frame.channels;
            int dstX = x + padX, dstY = y + padY;

            // BGR -> RGB, normalise to [0,1]
            out[0 * H * W + dstY * W + dstX] = frame.data[srcIdx + 2] / 255.f; // R
            out[1 * H * W + dstY * W + dstX] = frame.data[srcIdx + 1] / 255.f; // G
            out[2 * H * W + dstY * W + dstX] = frame.data[srcIdx + 0] / 255.f; // B
        }
    }
    return out;
}

// ─── Post-processing: parse YOLO output tensor, apply NMS ────────────────────
// Assumes YOLOv8 output layout: [batch, 84, 8400]
// 84 = 4 (cx,cy,w,h) + 80 class scores
std::vector<DetectedObject>
ObjectDetector::postprocess(
    const std::unordered_map<std::string, std::vector<float>>& raw,
    int origW, int origH) const
{
    std::vector<DetectedObject> result;
    if (raw.empty()) return result;  // stub mode

    const auto* it = raw.begin();
    const std::vector<float>& data = it->second;

    int numAnchors = 8400;
    int numAttrs   = 84;
    if (static_cast<int>(data.size()) < numAttrs * numAnchors) return result;

    // Scale letterbox back to original image coords
    float scaleX  = static_cast<float>(cfg_.inputWidth)  / origW;
    float scaleY  = static_cast<float>(cfg_.inputHeight) / origH;
    float scale   = std::min(scaleX, scaleY);
    float padX    = (cfg_.inputWidth  - origW * scale) / 2.f;
    float padY    = (cfg_.inputHeight - origH * scale) / 2.f;

    std::vector<BoundingBox2D> boxes;
    std::vector<float>         scores;
    std::vector<ObjectClass>   classes;

    for (int a = 0; a < numAnchors; ++a) {
        float maxScore = 0.f;
        int   maxClass = 0;
        for (int c = 4; c < numAttrs; ++c) {
            float s = data[c * numAnchors + a];
            if (s > maxScore) { maxScore = s; maxClass = c - 4; }
        }
        if (maxScore < cfg_.confidenceThreshold) continue;

        float cx = data[0 * numAnchors + a];
        float cy = data[1 * numAnchors + a];
        float w  = data[2 * numAnchors + a];
        float h  = data[3 * numAnchors + a];

        // Remove letterbox padding and undo scale
        float x = (cx - padX) / scale - w / (2.f * scale);
        float y = (cy - padY) / scale - h / (2.f * scale);
        w /= scale; h /= scale;

        // Clamp to image bounds
        x = std::max(0.f, x); y = std::max(0.f, y);
        w = std::min(w, static_cast<float>(origW) - x);
        h = std::min(h, static_cast<float>(origH) - y);

        boxes.push_back({x, y, w, h});
        scores.push_back(maxScore);
        classes.push_back(maxClass < 80 ? kCocoLabelMap[maxClass]
                                        : ObjectClass::Unknown);
    }

    auto kept = math::nms(boxes, scores, cfg_.nmsIouThreshold,
                          cfg_.confidenceThreshold);
    result.reserve(kept.size());
    for (size_t idx : kept) {
        DetectedObject obj;
        obj.id         = detectionCount_ * 10000 + idx;
        obj.bbox2d     = boxes[idx];
        obj.confidence = scores[idx];
        obj.classId    = classes[idx];
        obj.distance   = 0.f;   // filled by sensor fusion
        obj.stamp      = now();
        result.push_back(obj);
    }
    return result;
}

} // namespace adas::perception
