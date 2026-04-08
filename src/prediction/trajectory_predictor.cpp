#include "prediction/trajectory_predictor.hpp"
#include "common/logger.hpp"
#include "common/math_utils.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace adas::prediction {

TrajectoryPredictor::TrajectoryPredictor(PredictorConfig cfg)
    : cfg_(std::move(cfg)) {}

void TrajectoryPredictor::reset() { histories_.clear(); }

// ─── Update history buffers ───────────────────────────────────────────────────
void TrajectoryPredictor::update(const std::vector<DetectedObject>& objects) {
    for (const auto& obj : objects) {
        auto& hist = histories_[obj.id];
        hist.push_back(obj);
        if (hist.size() > cfg_.historyFrames)
            hist.erase(hist.begin());
    }
    // Remove histories for objects no longer tracked (basic cleanup)
    std::vector<uint64_t> toRemove;
    for (const auto& [id, h] : histories_) {
        bool found = false;
        for (const auto& o : objects) if (o.id == id) { found = true; break; }
        if (!found) toRemove.push_back(id);
    }
    for (auto id : toRemove) histories_.erase(id);
}

Trajectory TrajectoryPredictor::predict(uint64_t objectId) const {
    auto it = histories_.find(objectId);
    if (it == histories_.end()) return {};

    const auto& h = it->second;
    switch (cfg_.model) {
    case PredictionModel::ConstantVelocity: return cvPredict(h);
    case PredictionModel::ConstantTurnRate: return ctraPredict(h);
    case PredictionModel::PolynomialFit:    return polyPredict(h);
    }
    return cvPredict(h);
}

std::unordered_map<uint64_t, Trajectory> TrajectoryPredictor::predictAll() const {
    std::unordered_map<uint64_t, Trajectory> result;
    for (const auto& [id, _] : histories_)
        result[id] = predict(id);
    return result;
}

// ─── Constant-velocity model ─────────────────────────────────────────────────
Trajectory TrajectoryPredictor::cvPredict(const History& h) const {
    if (h.size() < 2) {
        if (h.empty()) return {};
        // Single point – predict stationary
        int steps = static_cast<int>(cfg_.horizonSeconds / cfg_.dtSeconds);
        Trajectory traj;
        for (int i = 1; i <= steps; ++i) {
            TrajectoryPoint tp;
            tp.position   = h.back().bbox3d.center;
            tp.speed      = h.back().velocity;
            tp.heading    = 0.f;
            tp.timeOffset = i * cfg_.dtSeconds;
            traj.push_back(tp);
        }
        return traj;
    }

    const auto& prev = h[h.size()-2];
    const auto& curr = h[h.size()-1];
    float vx = curr.bbox3d.center.x - prev.bbox3d.center.x;
    float vy = curr.bbox3d.center.y - prev.bbox3d.center.y;
    float speed   = std::sqrt(vx*vx + vy*vy);
    float heading = std::atan2(vy, vx);

    int steps = static_cast<int>(cfg_.horizonSeconds / cfg_.dtSeconds);
    Trajectory traj;
    traj.reserve(steps);

    float cx = curr.bbox3d.center.x;
    float cy = curr.bbox3d.center.y;
    for (int i = 1; i <= steps; ++i) {
        float dt = i * cfg_.dtSeconds;
        TrajectoryPoint tp;
        tp.position.x = cx + vx * dt;
        tp.position.y = cy + vy * dt;
        tp.position.z = curr.bbox3d.center.z;
        tp.speed      = speed;
        tp.heading    = heading;
        tp.timeOffset = dt;
        traj.push_back(tp);
    }
    return traj;
}

// ─── Constant turn-rate & acceleration (CTRA) ────────────────────────────────
Trajectory TrajectoryPredictor::ctraPredict(const History& h) const {
    if (h.size() < 3) return cvPredict(h);

    // Estimate yaw-rate from last 3 states
    auto angle2 = [](const DetectedObject& a, const DetectedObject& b) {
        return std::atan2(b.bbox3d.center.y - a.bbox3d.center.y,
                          b.bbox3d.center.x - a.bbox3d.center.x);
    };

    float hdg0  = angle2(h[h.size()-3], h[h.size()-2]);
    float hdg1  = angle2(h[h.size()-2], h[h.size()-1]);
    float yawRate = math::wrapAngle(hdg1 - hdg0);

    const auto& curr = h.back();
    float speed = curr.velocity > 0.f ? curr.velocity
                : math::euclidean3D(h[h.size()-2].bbox3d.center.x,
                                    h[h.size()-2].bbox3d.center.y, 0,
                                    curr.bbox3d.center.x,
                                    curr.bbox3d.center.y, 0);
    float heading = hdg1;
    float cx = curr.bbox3d.center.x;
    float cy = curr.bbox3d.center.y;

    int steps = static_cast<int>(cfg_.horizonSeconds / cfg_.dtSeconds);
    Trajectory traj;
    traj.reserve(steps);

    for (int i = 1; i <= steps; ++i) {
        float dt  = cfg_.dtSeconds;
        float dh  = yawRate * dt;
        if (std::abs(yawRate) < 1e-4f) {
            cx += speed * std::cos(heading) * dt;
            cy += speed * std::sin(heading) * dt;
        } else {
            cx += (speed / yawRate) * (std::sin(heading + dh) - std::sin(heading));
            cy += (speed / yawRate) * (std::cos(heading)       - std::cos(heading + dh));
        }
        heading = math::wrapAngle(heading + dh);

        TrajectoryPoint tp;
        tp.position.x  = cx;
        tp.position.y  = cy;
        tp.position.z  = curr.bbox3d.center.z;
        tp.speed       = speed;
        tp.heading     = heading;
        tp.timeOffset  = i * cfg_.dtSeconds;
        traj.push_back(tp);
    }
    return traj;
}

// ─── Polynomial (cubic spline) fit ───────────────────────────────────────────
Trajectory TrajectoryPredictor::polyPredict(const History& h) const {
    if (h.size() < 4) return ctraPredict(h);

    // Fit cubic polynomial to x(t) and y(t)
    // Use least-squares via normal equations
    int n = static_cast<int>(h.size());
    Eigen::MatrixXd A(n, 4);
    Eigen::VectorXd bx(n), by(n);

    for (int i = 0; i < n; ++i) {
        double t = static_cast<double>(i);
        A(i,0) = 1.0; A(i,1) = t; A(i,2) = t*t; A(i,3) = t*t*t;
        bx(i) = h[i].bbox3d.center.x;
        by(i) = h[i].bbox3d.center.y;
    }

    Eigen::VectorXd px = (A.transpose()*A).ldlt().solve(A.transpose()*bx);
    Eigen::VectorXd py = (A.transpose()*A).ldlt().solve(A.transpose()*by);

    int steps = static_cast<int>(cfg_.horizonSeconds / cfg_.dtSeconds);
    Trajectory traj;
    traj.reserve(steps);

    double tNow = static_cast<double>(n - 1);
    for (int i = 1; i <= steps; ++i) {
        double t = tNow + i;
        double t2 = t*t, t3 = t2*t;
        float x = static_cast<float>(px(0) + px(1)*t + px(2)*t2 + px(3)*t3);
        float y = static_cast<float>(py(0) + py(1)*t + py(2)*t2 + py(3)*t3);

        // Compute heading from derivative
        float dxdt = static_cast<float>(px(1) + 2*px(2)*t + 3*px(3)*t2);
        float dydt = static_cast<float>(py(1) + 2*py(2)*t + 3*py(3)*t2);

        TrajectoryPoint tp;
        tp.position.x  = x;
        tp.position.y  = y;
        tp.position.z  = h.back().bbox3d.center.z;
        tp.speed       = std::sqrt(dxdt*dxdt + dydt*dydt);
        tp.heading     = std::atan2(dydt, dxdt);
        tp.timeOffset  = i * cfg_.dtSeconds;
        traj.push_back(tp);
    }
    return traj;
}

} // namespace adas::prediction
