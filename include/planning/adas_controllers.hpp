#pragma once

#include "common/types.hpp"
#include "common/math_utils.hpp"
#include <optional>

namespace adas::planning {

// ─── ACC ─────────────────────────────────────────────────────────────────────
struct ACCConfig {
    float setSpeedMps{30.f};      // desired cruising speed (m/s)
    float minGapMetres{5.f};      // minimum following distance
    float timeHeadwaySec{1.8f};   // desired time gap to lead vehicle
    float maxAccel{2.0f};         // m/s²
    float maxDecel{3.5f};         // m/s² (comfortable decel, not AEB)
    // PID gains for speed control
    float kp{0.5f}, ki{0.02f}, kd{0.1f};
};

/// Adaptive Cruise Control.
/// Outputs throttle [0,1] and brake [0,1] to maintain safe headway
/// to the lead vehicle while targeting the set speed.
class ACCController {
public:
    explicit ACCController(ACCConfig cfg = {});

    /// @param ego         current ego state
    /// @param leadObject  closest in-path object, if any
    /// @returns VehicleControl with throttle/brake set
    VehicleControl update(const EgoState& ego,
                          const std::optional<DetectedObject>& leadObject,
                          double dt);

    void setSpeed(float mps) { cfg_.setSpeedMps = mps; }
    void enable(bool on)     { active_ = on; }
    bool isActive()   const  { return active_; }
    void reset();

    float desiredGap(float egoSpeed) const;

private:
    ACCConfig              cfg_;
    math::PID<double>      speedPid_;
    bool                   active_{true};
};

// ─── AEB ─────────────────────────────────────────────────────────────────────
struct AEBConfig {
    float ttcWarnSec{2.5f};    // TTC threshold for audio/visual warning
    float ttcPartialSec{1.8f}; // TTC for partial braking
    float ttcFullSec{1.2f};    // TTC for full autonomous braking
    float minObjectConfidence{0.5f};
    float maxBrakeForce{1.0f}; // [0,1] – full hydraulic braking
    float partialBrakeForce{0.4f};
};

enum class AEBState { Inactive, Warning, PartialBrake, FullBrake };

inline const char* toString(AEBState s) {
    static constexpr const char* names[] =
        {"Inactive","Warning","PartialBrake","FullBrake"};
    return names[static_cast<int>(s)];
}

/// Automatic Emergency Braking.
/// Evaluates the threat level of in-path objects and commands braking
/// before a collision becomes unavoidable.
class AEBController {
public:
    explicit AEBController(AEBConfig cfg = {});

    /// @param objects    all tracked objects in front of vehicle
    /// @param ego        current ego state
    /// @returns VehicleControl (brake field driven by AEB; others zeroed)
    VehicleControl update(const std::vector<DetectedObject>& objects,
                          const EgoState& ego);

    AEBState currentState() const { return state_; }

    /// TTC to the most critical object in the current cycle
    float criticalTTC() const { return criticalTtc_; }

    void  enable(bool on) { active_ = on; }
    bool  isActive() const { return active_; }

private:
    bool isInPath(const DetectedObject& obj, float egoSpeed) const;

    AEBConfig cfg_;
    AEBState  state_{AEBState::Inactive};
    float     criticalTtc_{std::numeric_limits<float>::max()};
    bool      active_{true};
};

// ─── LKA ─────────────────────────────────────────────────────────────────────
struct LKAConfig {
    float  maxLateralErrorM{0.3f};    // alert beyond this deviation
    float  maxSteeringRad{0.15f};     // max corrective steering angle
    float  minSpeedMps{15.f};         // LKA active above this speed
    // PID for lateral error correction
    float  kp{0.8f}, ki{0.01f}, kd{0.2f};
};

/// Lane Keeping Assist.
/// Uses lateral error and heading error from LaneDetector to compute
/// a corrective steering overlay.
class LaneKeepingAssist {
public:
    explicit LaneKeepingAssist(LKAConfig cfg = {});

    /// @param lane  output from LaneDetector
    /// @param ego   current ego state
    /// @param dt    time step
    /// @returns steering correction in radians (positive = left)
    VehicleControl update(const LaneInfo& lane, const EgoState& ego, double dt);

    bool  isActive() const { return active_; }
    void  enable(bool on)  { active_ = on; }
    void  reset();

private:
    LKAConfig         cfg_;
    math::PID<double> lateralPid_;
    bool              active_{true};
};

// ─── Traffic-Light / Stop-Line Handler ───────────────────────────────────────
struct TrafficLightHandlerConfig {
    float stopLineLookAheadM{30.f};
    float decelerationMss{2.5f};
};

class TrafficLightHandler {
public:
    explicit TrafficLightHandler(TrafficLightHandlerConfig cfg = {});

    /// Inspect detected objects for traffic lights; command stop if red.
    VehicleControl update(const std::vector<DetectedObject>& objects,
                          const EgoState& ego,
                          double dt);

    TrafficLightColor currentPhase() const { return phase_; }

private:
    TrafficLightHandlerConfig cfg_;
    TrafficLightColor         phase_{TrafficLightColor::Unknown};
    math::PID<double>         stopPid_;
};

} // namespace adas::planning
