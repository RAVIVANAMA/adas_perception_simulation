/// test_acc_controller.cpp
/// Unit tests for the Adaptive Cruise Control controller.
#include <gtest/gtest.h>
#include "planning/adas_controllers.hpp"

using namespace adas;
using namespace adas::planning;

static EgoState makeEgo(float speed) {
    EgoState e; e.speed = speed; e.stamp = now(); return e;
}
static DetectedObject makeLead(float dist, float vel) {
    DetectedObject o;
    o.classId = ObjectClass::Car;
    o.distance = dist;
    o.velocity = vel;
    o.bbox3d.center.x = dist;
    o.bbox3d.center.y = 0.f;
    o.stamp = now();
    return o;
}

// ─── Tests ───────────────────────────────────────────────────────────────────
TEST(ACCController, Throttle_WhenBelowSetSpeed) {
    ACCConfig cfg;
    cfg.setSpeedMps = 30.f;
    ACCController ctrl(cfg);
    auto ego = makeEgo(20.f);   // 10 m/s below set speed, no lead

    auto result = ctrl.update(ego, std::nullopt, 0.1);

    EXPECT_GT(result.throttle, 0.f);
    EXPECT_FLOAT_EQ(result.brake, 0.f);
    EXPECT_TRUE(result.accActive);
}

TEST(ACCController, Brake_WhenTooCloseToLead) {
    ACCConfig cfg;
    cfg.setSpeedMps    = 30.f;
    cfg.minGapMetres   = 5.f;
    cfg.timeHeadwaySec = 1.8f;
    ACCController ctrl(cfg);

    // Required gap at ego=30 m/s = 5 + 1.8*30 = 59 m
    // Actual gap = 10 m  →  should brake
    auto ego  = makeEgo(30.f);
    auto lead = makeLead(10.f, 5.f);  // lead doing 5 m/s

    auto result = ctrl.update(ego, lead, 0.1);

    EXPECT_GT(result.brake, 0.f);
    EXPECT_FLOAT_EQ(result.throttle, 0.f);
}

TEST(ACCController, Idle_WhenDisabled) {
    ACCController ctrl;
    ctrl.enable(false);
    auto ego = makeEgo(20.f);
    auto result = ctrl.update(ego, std::nullopt, 0.1);
    EXPECT_FLOAT_EQ(result.throttle, 0.f);
    EXPECT_FLOAT_EQ(result.brake,    0.f);
    EXPECT_FALSE(result.accActive);
}

TEST(ACCController, DesiredGap_ScalesWithSpeed) {
    ACCConfig cfg;
    cfg.minGapMetres   = 5.f;
    cfg.timeHeadwaySec = 2.0f;
    ACCController ctrl(cfg);

    EXPECT_FLOAT_EQ(ctrl.desiredGap(0.f),  5.f);
    EXPECT_FLOAT_EQ(ctrl.desiredGap(10.f), 25.f);
    EXPECT_FLOAT_EQ(ctrl.desiredGap(30.f), 65.f);
}

TEST(ACCController, SpeedLimitedByLead) {
    ACCConfig cfg;
    cfg.setSpeedMps = 30.f;
    ACCController ctrl(cfg);

    // Lead is 20 m ahead, doing 10 m/s.  Desired gap = 5 + 1.8*25 = 50m.
    // Since 20 < 50, ACC should not be commanding full throttle.
    auto ego  = makeEgo(25.f);
    auto lead = makeLead(20.f, 10.f);

    auto r1 = ctrl.update(ego, lead, 0.1);
    EXPECT_LE(r1.throttle, 0.5f);  // should be moderate or negative
}

TEST(ACCController, Cruises_WhenNoLead) {
    ACCConfig cfg;
    cfg.setSpeedMps = 30.f;
    ACCController ctrl(cfg);

    // Ego at set speed, no lead → maintain (near zero output)
    auto ego = makeEgo(30.f);
    auto result = ctrl.update(ego, std::nullopt, 0.1);

    // Should be roughly in balance – neither heavy throttle nor brake
    EXPECT_LT(result.throttle, 0.5f);
    EXPECT_LT(result.brake,    0.1f);
}
