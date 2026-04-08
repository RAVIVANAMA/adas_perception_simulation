#include "planning/adas_controllers.hpp"
#include "common/logger.hpp"

namespace adas::planning {

TrafficLightHandler::TrafficLightHandler(TrafficLightHandlerConfig cfg)
    : cfg_(std::move(cfg))
    , stopPid_(0.3, 0.02, 0.1, -1.0, 0.0)
{}

VehicleControl TrafficLightHandler::update(
    const std::vector<DetectedObject>& objects,
    const EgoState& ego,
    double dt)
{
    VehicleControl ctrl;
    phase_ = TrafficLightColor::Unknown;

    for (const auto& obj : objects) {
        if (obj.classId != ObjectClass::TrafficLight) continue;
        if (obj.distance > cfg_.stopLineLookAheadM)   continue;

        if (obj.tlColor == TrafficLightColor::Red
         || obj.tlColor == TrafficLightColor::Amber) {
            phase_ = obj.tlColor;
            if (ego.speed > 0.1f) {
                float  distToStop = std::max(0.f, obj.distance - 2.f);
                double brake = stopPid_.update(0.0, -static_cast<double>(distToStop), dt);
                ctrl.brake    = static_cast<float>(std::min(-brake, 1.0));
                ctrl.throttle = 0.f;
                LOG_DEBUG("TLH " << (obj.tlColor == TrafficLightColor::Red ? "RED" : "AMBER")
                          << " dist=" << obj.distance << "m brake=" << ctrl.brake);
            }
            return ctrl;
        }
        if (obj.tlColor == TrafficLightColor::Green)
            phase_ = TrafficLightColor::Green;
    }
    return ctrl;
}

} // namespace adas::planning
