#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

namespace adas {
namespace math {

// ─── Constants ────────────────────────────────────────────────────────────────
inline constexpr double PI  = 3.14159265358979323846;
inline constexpr double TAU = 2.0 * PI;
inline constexpr double DEG2RAD = PI / 180.0;
inline constexpr double RAD2DEG = 180.0 / PI;

// ─── Clamp (C++17 std::clamp alternative that works on any arithmetic type) ──
template<typename T,
         typename = std::enable_if_t<std::is_arithmetic_v<T>>>
constexpr T clamp(T val, T lo, T hi) noexcept {
    return val < lo ? lo : (val > hi ? hi : val);
}

// ─── Linear interpolation ─────────────────────────────────────────────────────
template<typename T>
constexpr T lerp(T a, T b, T t) noexcept {
    return a + t * (b - a);
}

// ─── Angle wrapping to (-π, π] ────────────────────────────────────────────────
template<typename T>
T wrapAngle(T angle) noexcept {
    while (angle >  static_cast<T>(PI)) angle -= static_cast<T>(TAU);
    while (angle <= -static_cast<T>(PI)) angle += static_cast<T>(TAU);
    return angle;
}

// ─── Distance helpers ─────────────────────────────────────────────────────────
inline float euclidean2D(float x1, float y1, float x2, float y2) noexcept {
    float dx = x2 - x1, dy = y2 - y1;
    return std::sqrt(dx*dx + dy*dy);
}

inline float euclidean3D(float x1,float y1,float z1,
                          float x2,float y2,float z2) noexcept {
    float dx=x2-x1, dy=y2-y1, dz=z2-z1;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// ─── Time-to-collision (TTC) ─────────────────────────────────────────────────
// Returns TTC in seconds; NaN if closing speed <= 0 (diverging/stationary)
inline float computeTTC(float distance_m, float closingSpeed_mps) noexcept {
    if (closingSpeed_mps <= 0.f) return std::numeric_limits<float>::quiet_NaN();
    return distance_m / closingSpeed_mps;
}

// ─── PID controller (stateful helper) ────────────────────────────────────────
template<typename T = double>
class PID {
public:
    PID(T kp, T ki, T kd, T outMin, T outMax)
        : kp_(kp), ki_(ki), kd_(kd)
        , outMin_(outMin), outMax_(outMax) {}

    T update(T setpoint, T measured, T dt) noexcept {
        T error    = setpoint - measured;
        integral_  = clamp<T>(integral_ + error * dt, outMin_ / ki_, outMax_ / ki_);
        T derivative = (dt > T{0}) ? (error - prevError_) / dt : T{0};
        prevError_ = error;
        T output   = kp_*error + ki_*integral_ + kd_*derivative;
        return clamp(output, outMin_, outMax_);
    }

    void reset() noexcept { integral_ = T{0}; prevError_ = T{0}; }

private:
    T kp_, ki_, kd_;
    T outMin_, outMax_;
    T integral_{};
    T prevError_{};
};

// ─── Simple 1-D Kalman filter ─────────────────────────────────────────────────
class KalmanFilter1D {
public:
    KalmanFilter1D(double processNoise, double measurementNoise, double estimate=0.0)
        : q_(processNoise), r_(measurementNoise), x_(estimate) {}

    double update(double measurement) noexcept {
        // Predict
        p_ += q_;
        // Update
        double k = p_ / (p_ + r_);
        x_ = x_ + k * (measurement - x_);
        p_ = (1.0 - k) * p_;
        return x_;
    }

    double value() const noexcept { return x_; }

private:
    double q_, r_;  // process & measurement noise
    double x_{0.0}, p_{1.0};
};

// ─── Intersection-over-Union (2-D axis-aligned boxes) ─────────────────────────
struct AABB { float x, y, w, h; };

inline float iou(const AABB& a, const AABB& b) noexcept {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x+a.w, b.x+b.w);
    float y2 = std::min(a.y+a.h, b.y+b.h);
    if (x2 <= x1 || y2 <= y1) return 0.f;
    float inter = (x2-x1)*(y2-y1);
    float uni   = a.w*a.h + b.w*b.h - inter;
    return (uni > 0.f) ? inter/uni : 0.f;
}

// ─── Non-Maximum Suppression ─────────────────────────────────────────────────
// Returns indices of surviving detections.
template<typename Box>
std::vector<size_t> nms(const std::vector<Box>& boxes,
                         const std::vector<float>& scores,
                         float iouThreshold,
                         float scoreThreshold = 0.f) {
    std::vector<size_t> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b){ return scores[a] > scores[b]; });

    std::vector<bool>   suppressed(boxes.size(), false);
    std::vector<size_t> kept;

    for (size_t i = 0; i < order.size(); ++i) {
        size_t idx = order[i];
        if (suppressed[idx] || scores[idx] < scoreThreshold) continue;
        kept.push_back(idx);
        for (size_t j = i + 1; j < order.size(); ++j) {
            size_t jdx = order[j];
            if (suppressed[jdx]) continue;
            AABB ba{boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height};
            AABB bb{boxes[jdx].x, boxes[jdx].y, boxes[jdx].width, boxes[jdx].height};
            if (iou(ba, bb) > iouThreshold) suppressed[jdx] = true;
        }
    }
    return kept;
}

} // namespace math
} // namespace adas
