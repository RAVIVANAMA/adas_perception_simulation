#include "planning/adas_controllers.hpp"
#include "common/logger.hpp"
#include "common/math_utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace adas::planning {

// ═══════════════════════════════════════════════════════════════════════════════
// Adaptive Cruise Control
// ═══════════════════════════════════════════════════════════════════════════════

ACCController::ACCController(ACCConfig cfg)
    : cfg_(std::move(cfg))
    , speedPid_(cfg_.kp, cfg_.ki, cfg_.kd,
                -cfg_.maxDecel, cfg_.maxAccel)
{}

void ACCController::reset() { speedPid_.reset(); }

float ACCController::desiredGap(float egoSpeed) const {
    return cfg_.minGapMetres + cfg_.timeHeadwaySec * egoSpeed;
}

VehicleControl ACCController::update(
    const EgoState&                     ego,
    const std::optional<DetectedObject>& lead,
    double dt)
{
    VehicleControl ctrl;
    if (!active_) return ctrl;

    float targetSpeed = cfg_.setSpeedMps;

    if (lead.has_value()) {
        float gap = lead->distance;
        float reqGap = desiredGap(ego.speed);

        // Space-based speed adjustment: if gap < desired, reduce speed target
        if (gap < reqGap) {
            float gapRatio = std::max(0.f, gap / reqGap);
            targetSpeed = std::min(targetSpeed,
                lead->velocity + (cfg_.setSpeedMps - lead->velocity) * gapRatio);
        }

        LOG_DEBUG("ACC: gap=" << gap << "m req=" << reqGap
                  << "m  lead_v=" << lead->velocity
                  << "m/s  target=" << targetSpeed << "m/s");
    }

    double pidOut = speedPid_.update(targetSpeed, ego.speed, dt);

    if (pidOut >= 0.0) {
        ctrl.throttle = static_cast<float>(std::min(pidOut / cfg_.maxAccel, 1.0));
        ctrl.brake    = 0.f;
    } else {
        ctrl.throttle = 0.f;
        ctrl.brake    = static_cast<float>(std::min(-pidOut / cfg_.maxDecel, 1.0));
    }
    ctrl.accActive = true;
    return ctrl;
}

} // namespace adas::planning
