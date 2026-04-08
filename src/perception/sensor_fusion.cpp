#include "perception/sensor_fusion.hpp"
#include "common/logger.hpp"
#include "common/math_utils.hpp"
#include <algorithm>
#include <limits>
#include <cmath>

namespace adas::perception {

// ─── EKF matrices ────────────────────────────────────────────────────────────
// State: [x, y, vx, vy, ax, ay]
static constexpr int kStateDim = 6;
static constexpr int kMeasDim  = 2;   // (x, y) position

static Eigen::MatrixXd makeF(double dt) {
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(kStateDim, kStateDim);
    // kinematics: x += vx*dt + 0.5*ax*dt²
    F(0,2) = dt;    F(0,4) = 0.5*dt*dt;
    F(1,3) = dt;    F(1,5) = 0.5*dt*dt;
    F(2,4) = dt;
    F(3,5) = dt;
    return F;
}

static Eigen::MatrixXd makeQ(double dt, double sigma) {
    // Discrete noise model (Singer / piecewise constant acceleration)
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(kStateDim, kStateDim);
    double dt2 = dt*dt, dt3 = dt2*dt, dt4 = dt3*dt;
    double q = sigma * sigma;
    Q(0,0) = Q(1,1) = dt4/4.0 * q;
    Q(2,2) = Q(3,3) = dt2     * q;
    Q(4,4) = Q(5,5) = 1.0     * q;
    Q(0,2) = Q(2,0) = Q(1,3) = Q(3,1) = dt3/2.0 * q;
    Q(0,4) = Q(4,0) = Q(1,5) = Q(5,1) = dt2/2.0 * q;
    Q(2,4) = Q(4,2) = Q(3,5) = Q(5,3) = dt      * q;
    return Q;
}

static Eigen::MatrixXd makeH() {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(kMeasDim, kStateDim);
    H(0,0) = 1.0;
    H(1,1) = 1.0;
    return H;
}

// ─── Constructor ──────────────────────────────────────────────────────────────
SensorFusion::SensorFusion(SensorFusionConfig cfg)
    : cfg_(std::move(cfg))
    , lastUpdateTime_(now())
{}

SensorFusion::~SensorFusion() = default;

void SensorFusion::reset() {
    tracks_.clear();
    nextId_ = 1;
    lastUpdateTime_ = now();
}

// ─── Main update ─────────────────────────────────────────────────────────────
void SensorFusion::update(
    const std::vector<DetectedObject>& camDets,
    const RadarFrame&                   radar,
    const LidarFrame*                   /*lidar*/,
    const EgoState&                     /*ego*/)
{
    auto t = now();
    double dt = std::chrono::duration<double>(t - lastUpdateTime_).count();
    dt = std::max(dt, 1e-4); // guard against zero dt
    lastUpdateTime_ = t;

    // ── 1. Predict all existing tracks ─────────────────────────────────────
    for (auto& [id, track] : tracks_) {
        predict(track, dt);
        ++track.age;
        ++track.lostFrames;
    }

    // ── 2. Associate camera detections → tracks ────────────────────────────
    auto camAssoc = associateCamera(camDets);
    for (const auto& [trackId, detIdx] : camAssoc) {
        auto& track = tracks_.at(trackId);
        updateCamera(track, camDets[detIdx]);
        track.lostFrames = 0;
        ++track.cameraHits;
        ++track.totalHits;
    }

    // ── 3. Spawn new tracks for unmatched detections ───────────────────────
    std::vector<bool> matched(camDets.size(), false);
    for (const auto& [tid, didx] : camAssoc) matched[didx] = true;

    for (size_t i = 0; i < camDets.size(); ++i) {
        if (matched[i]) continue;
        FusionTrack track;
        track.id    = nextId_++;
        track.state = Eigen::VectorXd::Zero(kStateDim);
        track.state(0) = camDets[i].bbox3d.center.x;
        track.state(1) = camDets[i].bbox3d.center.y;
        track.covariance = Eigen::MatrixXd::Identity(kStateDim, kStateDim) * 100.0;
        track.classId    = camDets[i].classId;
        track.confidence = camDets[i].confidence;
        track.lastBbox   = camDets[i].bbox2d;
        track.lastUpdate = t;
        tracks_[track.id] = std::move(track);
    }

    // ── 4. Fuse radar into nearest track ─────────────────────────────────
    for (const auto& tgt : radar.targets) {
        float tx = tgt.range * std::cos(tgt.azimuth);
        float ty = tgt.range * std::sin(tgt.azimuth);

        // Find closest track
        uint64_t bestId = 0;
        double   bestDist = cfg_.associationDistThreshold;
        for (auto& [id, track] : tracks_) {
            double dx = track.state(0) - tx;
            double dy = track.state(1) - ty;
            double d  = std::sqrt(dx*dx + dy*dy);
            if (d < bestDist) { bestDist = d; bestId = id; }
        }
        if (bestId != 0) {
            updateRadar(tracks_.at(bestId), tgt);
            ++tracks_.at(bestId).radarHits;
        }
    }

    // ── 5. Remove stale tracks ─────────────────────────────────────────────
    std::vector<uint64_t> toErase;
    for (const auto& [id, track] : tracks_) {
        if (track.lostFrames > cfg_.maxLostFrames) toErase.push_back(id);
    }
    for (auto id : toErase) {
        tracks_.erase(id);
        LOG_DEBUG("SensorFusion: deleted track " << id);
    }
}

// ─── EKF predict step ─────────────────────────────────────────────────────────
void SensorFusion::predict(FusionTrack& track, double dt) {
    auto F = makeF(dt);
    auto Q = makeQ(dt, cfg_.processNoise);
    track.state      = F * track.state;
    track.covariance = F * track.covariance * F.transpose() + Q;
}

// ─── EKF update – camera ──────────────────────────────────────────────────────
void SensorFusion::updateCamera(FusionTrack& track, const DetectedObject& det) {
    // Back-project bbox centre to approximate world-x,y using pinhole model
    float u = det.bbox2d.cx();
    float v = det.bbox2d.cy();
    float depth = (det.distance > 0.f) ? det.distance : track.state(0);
    float wx = (u - cfg_.cameraExtrinsic(0,3)) / cfg_.focalLength * depth;
    float wy = (v - cfg_.cameraExtrinsic(1,3)) / cfg_.focalLength * depth;

    Eigen::VectorXd z(kMeasDim);
    z << wx, wy;

    auto H = makeH();
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(kMeasDim, kMeasDim)
                        * (cfg_.cameraRNoise * cfg_.cameraRNoise);

    Eigen::VectorXd innov = z - H * track.state;
    Eigen::MatrixXd S     = H * track.covariance * H.transpose() + R;
    Eigen::MatrixXd K     = track.covariance * H.transpose() * S.inverse();

    track.state      = track.state      + K * innov;
    track.covariance = (Eigen::MatrixXd::Identity(kStateDim,kStateDim) - K*H)
                       * track.covariance;

    track.classId    = det.classId;
    track.confidence = det.confidence;
    track.lastBbox   = det.bbox2d;
    track.lastUpdate = now();
}

// ─── EKF update – radar ──────────────────────────────────────────────────────
void SensorFusion::updateRadar(FusionTrack& track, const RadarTarget& tgt) {
    float tx = tgt.range * std::cos(tgt.azimuth);
    float ty = tgt.range * std::sin(tgt.azimuth);

    Eigen::VectorXd z(kMeasDim);
    z << tx, ty;

    auto H = makeH();
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(kMeasDim, kMeasDim)
                        * (cfg_.radarRNoise * cfg_.radarRNoise);

    Eigen::VectorXd innov = z - H * track.state;
    Eigen::MatrixXd S     = H * track.covariance * H.transpose() + R;
    Eigen::MatrixXd K     = track.covariance * H.transpose() * S.inverse();

    track.state      += K * innov;
    track.covariance  = (Eigen::MatrixXd::Identity(kStateDim,kStateDim) - K*H)
                        * track.covariance;

    // Update velocity estimate from doppler
    float heading = std::atan2(ty, tx);
    track.state(2) = -tgt.rangeRate * std::cos(heading);
    track.state(3) = -tgt.rangeRate * std::sin(heading);
    track.lastUpdate = now();
}

// ─── Camera data association (IoU-based) ──────────────────────────────────────
std::unordered_map<uint64_t, SensorFusion::DetIdx>
SensorFusion::associateCamera(const std::vector<DetectedObject>& dets) {
    std::unordered_map<uint64_t, DetIdx> result;
    if (dets.empty() || tracks_.empty()) return result;

    // Build IoU cost matrix and apply greedy matching
    struct Match { uint64_t trackId; size_t detIdx; float iou; };
    std::vector<Match> candidates;

    for (const auto& [tid, track] : tracks_) {
        const BoundingBox2D& tb = track.lastBbox;
        for (size_t d = 0; d < dets.size(); ++d) {
            const BoundingBox2D& db = dets[d].bbox2d;
            math::AABB a{tb.x, tb.y, tb.width, tb.height};
            math::AABB b{db.x, db.y, db.width, db.height};
            float iouVal = math::iou(a, b);
            if (iouVal >= cfg_.associationIouThreshold)
                candidates.push_back({tid, d, iouVal});
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Match& a, const Match& b){ return a.iou > b.iou; });

    std::vector<bool> usedDets(dets.size(), false);
    std::vector<bool> usedTrack;
    // Build a "used" map keyed by track id
    std::unordered_map<uint64_t, bool> usedTracks;

    for (const auto& m : candidates) {
        if (usedTracks[m.trackId] || usedDets[m.detIdx]) continue;
        result[m.trackId]      = m.detIdx;
        usedTracks[m.trackId]  = true;
        usedDets[m.detIdx]     = true;
    }
    return result;
}

// ─── Accessors ────────────────────────────────────────────────────────────────
std::vector<DetectedObject> SensorFusion::getTrackedObjects() const {
    std::vector<DetectedObject> result;
    for (const auto& [id, track] : tracks_) {
        if (track.cameraHits + track.radarHits < cfg_.minHitsToConfirm) continue;

        DetectedObject obj;
        obj.id         = track.id;
        obj.classId    = track.classId;
        obj.confidence = track.confidence;
        obj.bbox2d     = track.lastBbox;
        obj.distance   = track.distance();
        obj.velocity   = track.speed();
        obj.bbox3d.center.x = static_cast<float>(track.state(0));
        obj.bbox3d.center.y = static_cast<float>(track.state(1));
        obj.stamp      = track.lastUpdate;
        result.push_back(obj);
    }
    return result;
}

const std::unordered_map<uint64_t, FusionTrack>& SensorFusion::tracks() const {
    return tracks_;
}

} // namespace adas::perception
