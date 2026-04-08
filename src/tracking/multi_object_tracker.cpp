#include "tracking/multi_object_tracker.hpp"
#include "tracking/hungarian.hpp"
#include "common/logger.hpp"
#include "common/math_utils.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace adas::tracking {

// ─── Kalman Filter helpers for SORT ──────────────────────────────────────────
// State: [cx, cy, w, h, vcx, vcy, vw, vh]
static constexpr int kS = 8;
static constexpr int kM = 4;  // measurement: [cx, cy, w, h]

static Eigen::MatrixXd makeF_sort(double dt) {
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(kS, kS);
    F(0,4) = dt; F(1,5) = dt; F(2,6) = dt; F(3,7) = dt;
    return F;
}

static Eigen::MatrixXd makeQ_sort(double q) {
    return Eigen::MatrixXd::Identity(kS, kS) * q;
}

static Eigen::MatrixXd makeH_sort() {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(kM, kS);
    H(0,0) = H(1,1) = H(2,2) = H(3,3) = 1.0;
    return H;
}

static Eigen::MatrixXd makeR_sort(double r) {
    return Eigen::MatrixXd::Identity(kM, kM) * r;
}

static Eigen::MatrixXd makeP0() {
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(kS, kS);
    P(4,4) = P(5,5) = P(6,6) = P(7,7) = 1000.0;
    return P;
}

// ─── Constructor ──────────────────────────────────────────────────────────────
MultiObjectTracker::MultiObjectTracker(MOTConfig cfg) : cfg_(std::move(cfg)) {}

void MultiObjectTracker::reset() {
    tracks_.clear();
    nextId_ = 1;
}

// ─── Internal: initialise a new track from a detection ───────────────────────
void MultiObjectTracker::initTrack(const DetectedObject& det) {
    Track t;
    t.id         = nextId_++;
    t.classId    = det.classId;
    t.confidence = det.confidence;
    t.bbox       = det.bbox2d;
    t.lastSeen   = det.stamp;
    t.age        = 1;
    t.totalHits  = 1;

    t.kfState = Eigen::VectorXd::Zero(kS);
    t.kfState(0) = det.bbox2d.cx();
    t.kfState(1) = det.bbox2d.cy();
    t.kfState(2) = det.bbox2d.width;
    t.kfState(3) = det.bbox2d.height;

    t.kfCov = makeP0();
    tracks_.push_back(std::move(t));
}

// ─── Internal: KF predict step ───────────────────────────────────────────────
void MultiObjectTracker::predictAll() {
    auto F = makeF_sort(cfg_.dt);
    auto Q = makeQ_sort(cfg_.qScale);
    for (auto& t : tracks_) {
        t.kfState = F * t.kfState;
        t.kfCov   = F * t.kfCov * F.transpose() + Q;
        // Update stored box from predicted state
        t.bbox.x      = t.kfState(0) - t.kfState(2) / 2.f;
        t.bbox.y      = t.kfState(1) - t.kfState(3) / 2.f;
        t.bbox.width  = static_cast<float>(t.kfState(2));
        t.bbox.height = static_cast<float>(t.kfState(3));
    }
}

// ─── Internal: KF update step for one track ─────────────────────────────────
void MultiObjectTracker::updateTrack(Track& t, const DetectedObject& det) {
    auto H = makeH_sort();
    auto R = makeR_sort(cfg_.rScale);

    Eigen::VectorXd z(kM);
    z << det.bbox2d.cx(), det.bbox2d.cy(), det.bbox2d.width, det.bbox2d.height;

    Eigen::VectorXd  innov = z - H * t.kfState;
    Eigen::MatrixXd  S     = H * t.kfCov * H.transpose() + R;
    Eigen::MatrixXd  K     = t.kfCov * H.transpose() * S.inverse();

    t.kfState += K * innov;
    t.kfCov    = (Eigen::MatrixXd::Identity(kS, kS) - K * H) * t.kfCov;

    t.bbox       = det.bbox2d;
    t.classId    = det.classId;
    t.confidence = det.confidence;
    t.lastSeen   = det.stamp;
    ++t.totalHits;
    t.lostFrames = 0;
}

// ─── Main update ─────────────────────────────────────────────────────────────
std::vector<Track> MultiObjectTracker::update(
    const std::vector<DetectedObject>& detections)
{
    // 1. Predict
    predictAll();
    for (auto& t : tracks_) { ++t.age; ++t.lostFrames; }

    // 2. Build IoU cost matrix
    int nT = static_cast<int>(tracks_.size());
    int nD = static_cast<int>(detections.size());

    std::vector<int> assignment;
    if (nT > 0 && nD > 0) {
        std::vector<std::vector<double>> cost(nT, std::vector<double>(nD));
        for (int i = 0; i < nT; ++i) {
            for (int j = 0; j < nD; ++j) {
                const BoundingBox2D& tb = tracks_[i].bbox;
                const BoundingBox2D& db = detections[j].bbox2d;
                math::AABB ta{tb.x,tb.y,tb.width,tb.height};
                math::AABB da{db.x,db.y,db.width,db.height};
                cost[i][j] = 1.0 - math::iou(ta, da);  // min cost = max IoU
            }
        }
        assignment = solveHungarian(cost);
    } else {
        assignment.assign(nT, -1);
    }

    // 3. Apply assignments
    std::vector<bool> detUsed(nD, false);
    for (int i = 0; i < nT; ++i) {
        int j = (i < static_cast<int>(assignment.size())) ? assignment[i] : -1;
        if (j >= 0 && j < nD) {
            const BoundingBox2D& tb = tracks_[i].bbox;
            const BoundingBox2D& db = detections[j].bbox2d;
            math::AABB ta{tb.x,tb.y,tb.width,tb.height};
            math::AABB da{db.x,db.y,db.width,db.height};
            if (math::iou(ta,da) >= cfg_.iouThreshold) {
                updateTrack(tracks_[i], detections[j]);
                detUsed[j] = true;
            }
        }
    }

    // 4. Spawn tracks for unmatched detections
    for (int j = 0; j < nD; ++j) {
        if (!detUsed[j]) initTrack(detections[j]);
    }

    // 5. Remove dead tracks
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
            [&](const Track& t){ return t.lostFrames > cfg_.maxAge; }),
        tracks_.end());

    // 6. Return confirmed tracks
    std::vector<Track> result;
    for (const auto& t : tracks_) {
        if (t.totalHits >= cfg_.minHits && t.lostFrames == 0)
            result.push_back(t);
    }

    LOG_DEBUG("MOT: det=" << nD << " tracks=" << tracks_.size()
              << " confirmed=" << result.size());
    return result;
}

} // namespace adas::tracking
