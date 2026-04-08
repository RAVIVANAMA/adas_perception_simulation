#include "planning/adas_controllers.hpp"
#include "common/logger.hpp"
#include <cmath>

namespace adas::planning {

LaneKeepingAssist::LaneKeepingAssist(LKAConfig cfg)
    : cfg_(std::move(cfg))
    , lateralPid_(cfg_.kp, cfg_.ki, cfg_.kd,
                  -cfg_.maxSteeringRad, cfg_.maxSteeringRad)
{}

void LaneKeepingAssist::reset() { lateralPid_.reset(); }

VehicleControl LaneKeepingAssist::update(
    const LaneInfo& lane, const EgoState& ego, double dt)
{
    VehicleControl ctrl;
    if (!active_ || ego.speed < cfg_.minSpeedMps) return ctrl;
    if (!lane.left && !lane.right) return ctrl;

    // Combined error: lateral deviation + heading preview term
    double combinedError = lane.lateralError + 0.5 * lane.headingError;
    ctrl.steering  = static_cast<float>(lateralPid_.update(0.0, combinedError, dt));
    ctrl.lkaActive = true;

    if (lane.isDeparting) {
        LOG_WARN("LKA: departure detected  lat=" << lane.lateralError
                 << "m  hdg=" << lane.headingError
                 << "rad  steer=" << ctrl.steering);
    }
    return ctrl;
}

} // namespace adas::planning
