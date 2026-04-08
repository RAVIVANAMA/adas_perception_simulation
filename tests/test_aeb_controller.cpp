/// test_aeb_controller.cpp
/// Unit tests for the Automatic Emergency Braking controller.
#include <gtest/gtest.h>
#include "planning/adas_controllers.hpp"

using namespace adas;
using namespace adas::planning;

// ─── Helpers ─────────────────────────────────────────────────────────────────
static EgoState makeEgo(float speed) {
    EgoState e;
    e.speed = speed;
    e.stamp = now();
    return e;
}

static DetectedObject makeObject(float dist, float vel,
                                  float lat = 0.f, float conf = 0.9f) {
    DetectedObject o;
    o.classId          = ObjectClass::Car;
    o.confidence       = conf;
    o.distance         = dist;
    o.velocity         = vel;
    o.bbox3d.center.x  = dist;   // directly ahead
    o.bbox3d.center.y  = lat;
    o.stamp            = now();
    return o;
}

// ─── Tests ───────────────────────────────────────────────────────────────────
TEST(AEBController, Inactive_WhenDisabled) {
    AEBController ctrl;
    ctrl.enable(false);

    auto ego = makeEgo(25.f);
    auto obj = makeObject(10.f, 0.f);   // very close

    auto result = ctrl.update({obj}, ego);

    EXPECT_EQ(ctrl.currentState(), AEBState::Inactive);
    EXPECT_FLOAT_EQ(result.brake, 0.f);
    EXPECT_FALSE(result.aebActive);
}

TEST(AEBController, Warning_WhenTTCAbovePartialThreshold) {
    AEBController ctrl;
    // TTC = 20 m / 10 m/s = 2.0 s  →  between ttcWarnSec(2.5) and ttcPartialSec(1.8)
    auto ego = makeEgo(25.f);
    auto obj = makeObject(20.f, 15.f);  // closing speed = 10 m/s

    auto result = ctrl.update({obj}, ego);

    EXPECT_EQ(ctrl.currentState(), AEBState::Warning);
    EXPECT_FLOAT_EQ(result.brake, 0.f);
    EXPECT_FALSE(result.aebActive);
}

TEST(AEBController, PartialBrake_WhenTTCBelowThreshold) {
    AEBController ctrl;
    // TTC = 15 m / 10 m/s = 1.5 s  →  between 1.8 and 1.2
    auto ego = makeEgo(25.f);
    auto obj = makeObject(15.f, 15.f);

    auto result = ctrl.update({obj}, ego);

    EXPECT_EQ(ctrl.currentState(), AEBState::PartialBrake);
    EXPECT_GT(result.brake, 0.f);
    EXPECT_TRUE(result.aebActive);
}

TEST(AEBController, FullBrake_WhenImminent) {
    AEBController ctrl;
    // TTC = 5 m / 20 m/s = 0.25 s  →  below ttcFullSec(1.2)
    auto ego = makeEgo(25.f);
    auto obj = makeObject(5.f, 5.f);   // closing at 20 m/s

    auto result = ctrl.update({obj}, ego);

    EXPECT_EQ(ctrl.currentState(), AEBState::FullBrake);
    EXPECT_FLOAT_EQ(result.brake, 1.0f);
    EXPECT_FLOAT_EQ(result.throttle, 0.f);
    EXPECT_TRUE(result.aebActive);
}

TEST(AEBController, NoBrake_WhenObjectOutOfPath) {
    AEBController ctrl;
    // Object is far to the side (5 m lateral) – should not trigger
    auto ego = makeEgo(25.f);
    auto obj = makeObject(5.f, 0.f, 5.f);  // 5 m lateral offset

    auto result = ctrl.update({obj}, ego);

    EXPECT_EQ(ctrl.currentState(), AEBState::Inactive);
    EXPECT_FLOAT_EQ(result.brake, 0.f);
}

TEST(AEBController, NoBrake_WhenObjectBehind) {
    auto ego = makeEgo(25.f);
    DetectedObject obj = makeObject(10.f, 15.f);
    obj.bbox3d.center.x = -10.f;   // behind vehicle

    AEBController ctrl;
    auto result = ctrl.update({obj}, ego);

    EXPECT_EQ(ctrl.currentState(), AEBState::Inactive);
}

TEST(AEBController, NoBrake_LowConfidence) {
    AEBController ctrl;
    auto ego = makeEgo(25.f);
    auto obj = makeObject(5.f, 5.f, 0.f, 0.2f);  // low confidence

    auto result = ctrl.update({obj}, ego);
    EXPECT_EQ(ctrl.currentState(), AEBState::Inactive);
}

TEST(AEBController, MultipleObjects_MostCriticalDrives) {
    AEBController ctrl;
    auto ego = makeEgo(25.f);

    DetectedObject safe = makeObject(80.f, 0.f);  // no threat
    DetectedObject crit = makeObject(5.f,  5.f);  // imminent

    auto result = ctrl.update({safe, crit}, ego);
    EXPECT_EQ(ctrl.currentState(), AEBState::FullBrake);
}
