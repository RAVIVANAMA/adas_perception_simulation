#pragma once

#include "common/types.hpp"
#include "common/math_utils.hpp"
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

namespace adas::tracking {

// ─── Kalman-filtered track (image-space + world-space hybrid) ────────────────
// State: [cx, cy, width, height, vx, vy, vw, vh]  (8-D, SORT-style)
struct Track {
    uint64_t      id{0};
    Eigen::VectorXd kfState;      // 8-D Kalman state
    Eigen::MatrixXd kfCov;        // 8×8 covariance
    BoundingBox2D bbox;           // current predicted box
    ObjectClass   classId{ObjectClass::Unknown};
    float         confidence{0.f};
    int           age{0};
    int           totalHits{0};
    int           lostFrames{0};
    Timestamp     lastSeen;
};

// ─── Multi-Object Tracker configuration ──────────────────────────────────────
struct MOTConfig {
    float  iouThreshold{0.3f};       // min IoU for association
    int    minHits{3};               // min detections to confirm a track
    int    maxAge{5};                // max frames without matching
    double dt{0.033};                // nominal frame interval (s)
    // Kalman noise parameters
    double qScale{0.01};             // process noise scale
    double rScale{1.0};              // measurement noise scale
};

/// SORT (Simple Online and Realtime Tracking) with Kalman Filter
/// and Hungarian algorithm data association.
class MultiObjectTracker {
public:
    explicit MultiObjectTracker(MOTConfig cfg = {});

    /// Update tracker with new detections.
    /// @return confirmed tracks (age >= minHits, lostFrames == 0)
    std::vector<Track> update(const std::vector<DetectedObject>& detections);

    const std::vector<Track>& allTracks()  const { return tracks_; }
    size_t                    trackCount() const { return tracks_.size(); }
    void                      reset();

private:
    void   initTrack(const DetectedObject& det);
    void   predictAll();
    void   updateTrack(Track& t, const DetectedObject& det);

    MOTConfig            cfg_;
    std::vector<Track>   tracks_;
    uint64_t             nextId_{1};
};

} // namespace adas::tracking
