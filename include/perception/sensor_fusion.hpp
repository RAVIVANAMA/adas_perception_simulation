#pragma once

#include "common/types.hpp"
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include <memory>
#include <optional>

namespace adas::perception {

// ─── Extended Kalman Filter track state ──────────────────────────────────────
// State vector: [x, y, vx, vy, ax, ay]  (6-D, world frame, metres / m·s⁻¹)
struct FusionTrack {
    uint64_t      id{0};
    Eigen::VectorXd state;      // 6-D state estimate
    Eigen::MatrixXd covariance; // 6×6 estimate covariance
    ObjectClass   classId{ObjectClass::Unknown};
    float         confidence{0.f};
    int           cameraHits{0};
    int           radarHits{0};
    int           lidarHits{0};
    int           age{0};        // frames since creation
    int           lostFrames{0}; // frames without measurement
    Timestamp     lastUpdate;
    BoundingBox2D lastBbox;

    float speed() const {
        return static_cast<float>(
            std::sqrt(state(2)*state(2) + state(3)*state(3)));
    }
    float distance() const {
        return static_cast<float>(
            std::sqrt(state(0)*state(0) + state(1)*state(1)));
    }
};

// ─── Sensor fusion configuration ─────────────────────────────────────────────
struct SensorFusionConfig {
    // Noise params
    double processNoise{0.1};
    double cameraRNoise{2.0};   // pixel-level uncertainty mapped to metres
    double radarRNoise{0.3};    // m
    double lidarRNoise{0.15};   // m

    // Track management
    int   minHitsToConfirm{3};
    int   maxLostFrames{5};
    float associationIouThreshold{0.3f};   // camera association
    float associationDistThreshold{2.5f};  // m  (radar / lidar)

    // Sensor extrinsics (rotation + translation from sensor to vehicle frame)
    Eigen::Matrix4d cameraExtrinsic{Eigen::Matrix4d::Identity()};
    Eigen::Matrix4d radarExtrinsic{Eigen::Matrix4d::Identity()};
    Eigen::Matrix4d lidarExtrinsic{Eigen::Matrix4d::Identity()};

    // Camera intrinsics (for back-projection)
    float focalLength{800.f};
    float cx{640.f}, cy{360.f};
};

/// Late-fusion EKF that combines camera detections, radar targets, and
/// lidar clusters into a unified set of tracked objects.
class SensorFusion {
public:
    explicit SensorFusion(SensorFusionConfig cfg = {});
    ~SensorFusion();

    /// Process one set of synchronised sensor readings.
    /// @param camDets   detections from object detector (may be empty)
    /// @param radar     radar frame (may be empty)
    /// @param lidar     lidar frame (may be nullptr)
    /// @param ego       current ego-vehicle state
    void update(const std::vector<DetectedObject>& camDets,
                const RadarFrame&                   radar,
                const LidarFrame*                   lidar,
                const EgoState&                     ego);

    /// Return all currently confirmed tracks as DetectedObject structs.
    std::vector<DetectedObject> getTrackedObjects() const;

    /// Access raw fusion tracks (e.g. for testing / visualisation).
    const std::unordered_map<uint64_t, FusionTrack>& tracks() const;

    void reset();

private:
    // EKF predict step for one track over dt seconds
    void predict(FusionTrack& track, double dt);

    // EKF update step (camera)
    void updateCamera(FusionTrack& track, const DetectedObject& det);

    // EKF update step (radar)
    void updateRadar(FusionTrack& track, const RadarTarget& tgt);

    // Data association (Euclidean / IoU)
    using DetIdx   = size_t;
    using TrackIdx = uint64_t;
    std::unordered_map<uint64_t, DetIdx>
    associateCamera(const std::vector<DetectedObject>& dets);

    SensorFusionConfig                             cfg_;
    std::unordered_map<uint64_t, FusionTrack>      tracks_;
    uint64_t                                       nextId_{1};
    Timestamp                                      lastUpdateTime_;
};

} // namespace adas::perception
