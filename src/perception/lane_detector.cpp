#include "perception/lane_detector.hpp"
#include "common/logger.hpp"
#include "common/math_utils.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace adas::perception {

// ─── Constructor ──────────────────────────────────────────────────────────────
LaneDetector::LaneDetector(LaneDetectorConfig cfg)
    : cfg_(std::move(cfg))
{
    runner_ = inference::makeRunner(cfg_.backend, cfg_.deviceId);
    if (!cfg_.modelPath.empty()) {
        runner_->loadModel(cfg_.modelPath);
    }
    LOG_INFO("LaneDetector: ready (model='" << cfg_.modelPath << "')");
}

LaneDetector::~LaneDetector() = default;

bool LaneDetector::isReady() const {
    return runner_ && runner_->isLoaded();
}

// ─── Public API ───────────────────────────────────────────────────────────────
LaneInfo LaneDetector::detect(const CameraFrame& frame) {
    if (!isReady()) {
        LOG_WARN("LaneDetector::detect called but model not loaded");
        return {};
    }

    auto input = preprocess(frame);

    std::unordered_map<std::string, std::vector<float>> ins, outs;
    ins["input"] = std::move(input);
    runner_->run(ins, outs);

    return postprocess(outs, frame.width, frame.height);
}

// ─── Pre-processing ───────────────────────────────────────────────────────────
std::vector<float> LaneDetector::preprocess(const CameraFrame& frame) const {
    int H = cfg_.inputHeight, W = cfg_.inputWidth;
    std::vector<float> out(3 * H * W, 0.f);

    float scaleX = static_cast<float>(W) / frame.width;
    float scaleY = static_cast<float>(H) / frame.height;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int srcX = std::min(static_cast<int>(x / scaleX), frame.width  - 1);
            int srcY = std::min(static_cast<int>(y / scaleY), frame.height - 1);
            int si   = (srcY * frame.width + srcX) * frame.channels;

            // ImageNet normalisation: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
            out[0 * H * W + y * W + x] = (frame.data[si+2]/255.f - 0.485f) / 0.229f;
            out[1 * H * W + y * W + x] = (frame.data[si+1]/255.f - 0.456f) / 0.224f;
            out[2 * H * W + y * W + x] = (frame.data[si+0]/255.f - 0.406f) / 0.225f;
        }
    }
    return out;
}

// ─── Post-processing ──────────────────────────────────────────────────────────
// Stub implementation: uses output probability map to fit polynomials.
// In a real system this would decode UFLD row-anchor or SCNN segmentation output.
LaneInfo LaneDetector::postprocess(
    const std::unordered_map<std::string, std::vector<float>>& raw,
    int origW, int origH) const
{
    LaneInfo info;
    if (raw.empty()) return info; // stub mode

    const auto& data = raw.begin()->second;
    int H = cfg_.inputHeight, W = cfg_.inputWidth;
    if (static_cast<int>(data.size()) < H * W) return info;

    // Threshold segmentation mask into lane pixels, separate left/right by x < W/2
    LaneBoundary left, right;
    left.type  = LaneType::Solid;
    right.type = LaneType::Solid;

    float scaleX = static_cast<float>(origW) / W;
    float scaleY = static_cast<float>(origH) / H;

    for (int y = H / 2; y < H; ++y) {
        float maxLeft = 0.f, maxRight = 0.f;
        int   bestLeft = -1, bestRight = -1;

        for (int x = 0; x < W; ++x) {
            float p = data[y * W + x];
            if (x < W / 2) {
                if (p > maxLeft && p > cfg_.confidenceThreshold)
                    { maxLeft = p; bestLeft = x; }
            } else {
                if (p > maxRight && p > cfg_.confidenceThreshold)
                    { maxRight = p; bestRight = x; }
            }
        }
        if (bestLeft  >= 0)
            left.points.push_back({bestLeft  * scaleX, y * scaleY});
        if (bestRight >= 0)
            right.points.push_back({bestRight * scaleX, y * scaleY});
    }

    if (left.points.size() > 5)  info.left  = left;
    if (right.points.size() > 5) info.right = right;

    if (info.left && info.right)
        info = computeErrors(*info.left, *info.right, origW, origH);

    return info;
}

// ─── Static helper: compute lateral / heading errors ─────────────────────────
LaneInfo LaneDetector::computeErrors(
    const LaneBoundary& left, const LaneBoundary& right,
    int imageWidth, int imageHeight)
{
    LaneInfo info;
    info.left  = left;
    info.right = right;

    if (left.points.empty() || right.points.empty()) return info;

    // Use the bottom-most point from each boundary to estimate lane centre
    auto botLeft  = left.points.front();
    auto botRight = right.points.front();
    for (const auto& p : left.points)  if (p.y > botLeft.y)  botLeft  = p;
    for (const auto& p : right.points) if (p.y > botRight.y) botRight = p;

    float centreLaneX = (botLeft.x + botRight.x) / 2.f;
    float imageCentreX = imageWidth / 2.f;

    // Pixel → metres conversion (rough: assume 640px wide lane ≈ 3.7m)
    float pixelsPerMetre = (botRight.x - botLeft.x) / 3.7f;
    info.lateralError = (imageCentreX - centreLaneX) / pixelsPerMetre;

    // Heading error: angle of the lane midline near the bottom of image
    if (left.points.size() > 1 && right.points.size() > 1) {
        // Linear regression slope of mid-points
        std::vector<float> xs, ys;
        for (size_t i = 0; i < std::min(left.points.size(), right.points.size()); ++i) {
            xs.push_back((left.points[i].x + right.points[i].x) / 2.f);
            ys.push_back((left.points[i].y + right.points[i].y) / 2.f);
        }
        float n = static_cast<float>(xs.size());
        float sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
        for (size_t i = 0; i < xs.size(); ++i) {
            sumX  += xs[i]; sumY  += ys[i];
            sumXX += xs[i]*xs[i]; sumXY += xs[i]*ys[i];
        }
        float denom = n * sumXX - sumX * sumX;
        if (std::abs(denom) > 1e-6f) {
            float slope = (n * sumXY - sumX * sumY) / denom;
            // slope is dy/dx in image; lane heading error = atan(1/slope) - pi/2
            info.headingError = std::atan2(1.f, slope) - static_cast<float>(math::PI / 2.0);
            info.headingError = math::wrapAngle(info.headingError);
        }
    }

    info.isDeparting = std::abs(info.lateralError) > 0.3f
                    || std::abs(info.headingError)  > 0.1f;
    return info;
}

} // namespace adas::perception
