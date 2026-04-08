#include "planning/adas_controllers.hpp"
#include "common/logger.hpp"
#include "common/math_utils.hpp"
#include <cmath>
#include <limits>

namespace adas::planning {

AEBController::AEBController(AEBConfig cfg) : cfg_(std::move(cfg)) {}

bool AEBController::isInPath(const DetectedObject& obj, float /*egoSpeed*/) const {
    float lateralPos = obj.bbox3d.center.y;
    float longPos    = obj.bbox3d.center.x;
    return (longPos > 0.5f) && (std::abs(lateralPos) < 1.8f);
}

VehicleControl AEBController::update(
    const std::vector<DetectedObject>& objects,
    const EgoState& ego)
{
    VehicleControl ctrl;
    if (!active_) { state_ = AEBState::Inactive; return ctrl; }

    float minTTC = std::numeric_limits<float>::max();
    for (const auto& obj : objects) {
        if (obj.confidence < cfg_.minObjectConfidence) continue;
        if (!isInPath(obj, ego.speed)) continue;

        float closingSpeed = ego.speed - obj.velocity;
        float ttc = math::computeTTC(obj.distance, closingSpeed);
        if (!std::isnan(ttc)) minTTC = std::min(minTTC, ttc);
    }
    criticalTtc_ = minTTC;

    AEBState newState  = AEBState::Inactive;
    float    brakeCmd  = 0.f;

    if      (minTTC < cfg_.ttcFullSec)    { newState = AEBState::FullBrake;    brakeCmd = cfg_.maxBrakeForce; }
    else if (minTTC < cfg_.ttcPartialSec) { newState = AEBState::PartialBrake; brakeCmd = cfg_.partialBrakeForce; }
    else if (minTTC < cfg_.ttcWarnSec)    { newState = AEBState::Warning;      brakeCmd = 0.f; }

    if (newState != state_) {
        LOG_INFO("AEB: " << toString(state_) << " -> " << toString(newState)
                 << "  TTC=" << (minTTC < 1e9f ? minTTC : -1.f) << "s");
    }
    state_ = newState;

    ctrl.brake     = brakeCmd;
    ctrl.throttle  = (brakeCmd > 0.f) ? 0.f : ctrl.throttle;
    ctrl.aebActive = (state_ != AEBState::Inactive);
    return ctrl;
}

} // namespace adas::planning
