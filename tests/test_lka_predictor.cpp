/// test_lka_predictor.cpp
/// Unit tests for Lane Keeping Assist and Trajectory Predictor.
#include <gtest/gtest.h>
#include "planning/adas_controllers.hpp"
#include "prediction/trajectory_predictor.hpp"
#include "perception/lane_detector.hpp"

using namespace adas;
using namespace adas::planning;
using namespace adas::prediction;

// ─── LKA helpers ─────────────────────────────────────────────────────────────
static EgoState makeEgo(float speed) {
    EgoState e; e.speed = speed; e.stamp = now(); return e;
}

static LaneInfo makeLane(float lat, float hdg, bool departing = false) {
    LaneInfo l;
    l.lateralError = lat;
    l.headingError = hdg;
    l.isDeparting  = departing;
    LaneBoundary lb;
    lb.type = LaneType::Solid;
    lb.points.push_back({200.f, 400.f});
    l.left = l.right = lb;
    return l;
}

// ─── LKA tests ───────────────────────────────────────────────────────────────
TEST(LKA, NoCorrection_WhenDisabled) {
    LaneKeepingAssist lka;
    lka.enable(false);
    auto result = lka.update(makeLane(0.5f, 0.1f), makeEgo(25.f), 0.033);
    EXPECT_FLOAT_EQ(result.steering, 0.f);
    EXPECT_FALSE(result.lkaActive);
}

TEST(LKA, NoCorrection_WhenBelowMinSpeed) {
    LKAConfig cfg;
    cfg.minSpeedMps = 20.f;
    LaneKeepingAssist lka(cfg);
    auto result = lka.update(makeLane(0.5f, 0.1f), makeEgo(5.f), 0.033);
    EXPECT_FLOAT_EQ(result.steering, 0.f);
}

TEST(LKA, Correction_WhenDeparting) {
    LKAConfig cfg;
    cfg.minSpeedMps = 15.f;
    LaneKeepingAssist lka(cfg);
    // Positive lateral error → vehicle drifted right → need left correction
    auto result = lka.update(makeLane(0.4f, 0.0f, true), makeEgo(25.f), 0.033);
    EXPECT_TRUE(result.lkaActive);
    EXPECT_NE(result.steering, 0.f);
}

TEST(LKA, Steering_Bounded) {
    LKAConfig cfg;
    cfg.maxSteeringRad = 0.1f;
    cfg.minSpeedMps    = 0.f;
    LaneKeepingAssist lka(cfg);

    // Large lateral error
    for (int i = 0; i < 20; ++i)
        lka.update(makeLane(2.f, 0.5f), makeEgo(25.f), 0.033);

    auto result = lka.update(makeLane(2.f, 0.5f), makeEgo(25.f), 0.033);
    EXPECT_LE(std::abs(result.steering), cfg.maxSteeringRad + 1e-5f);
}

TEST(LKA, NoLaneBoundaries_NoCorrection) {
    LaneKeepingAssist lka;
    LaneInfo noLane;
    auto result = lka.update(noLane, makeEgo(25.f), 0.033);
    EXPECT_FLOAT_EQ(result.steering, 0.f);
}

// ─── LaneDetector static helper ───────────────────────────────────────────────
TEST(LaneDetector, ComputeErrors_SymmetricLane) {
    using adas::perception::LaneBoundary;
    LaneBoundary left, right;
    left.type  = LaneType::Solid;
    right.type = LaneType::Solid;
    for (int y = 240; y <= 480; y += 20) {
        left.points.push_back({160.f, static_cast<float>(y)});
        right.points.push_back({480.f, static_cast<float>(y)});
    }
    // Symmetric lane → lateral error should be near 0
    auto info = adas::perception::LaneDetector::computeErrors(left, right, 640, 480);
    EXPECT_NEAR(info.lateralError, 0.f, 0.1f);
}

// ─── Trajectory predictor tests ──────────────────────────────────────────────
static DetectedObject makeObj(uint64_t id, float x, float y, float v = 5.f) {
    DetectedObject o;
    o.id             = id;
    o.classId        = ObjectClass::Car;
    o.velocity       = v;
    o.bbox3d.center.x = x;
    o.bbox3d.center.y = y;
    o.stamp          = now();
    return o;
}

TEST(TrajectoryPredictor, EmptyHistory_EmptyTrajectory) {
    TrajectoryPredictor predictor;
    auto traj = predictor.predict(999);
    EXPECT_TRUE(traj.empty());
}

TEST(TrajectoryPredictor, CV_Predicts_ForwardMotion) {
    PredictorConfig cfg;
    cfg.model         = PredictionModel::ConstantVelocity;
    cfg.horizonSeconds = 1.0f;
    cfg.dtSeconds      = 0.1f;
    TrajectoryPredictor predictor(cfg);

    // Object moving in +X direction
    for (int i = 0; i < 5; ++i) {
        predictor.update({makeObj(1, static_cast<float>(i)*0.5f, 0.f)});
    }
    auto traj = predictor.predict(1);
    ASSERT_FALSE(traj.empty());
    // Last point should be further in +X than starting position
    EXPECT_GT(traj.back().position.x, 2.f);
}

TEST(TrajectoryPredictor, CTRA_Predicts_CurvedPath) {
    PredictorConfig cfg;
    cfg.model          = PredictionModel::ConstantTurnRate;
    cfg.horizonSeconds = 2.f;
    cfg.dtSeconds      = 0.1f;
    cfg.historyFrames  = 10;
    TrajectoryPredictor predictor(cfg);

    // Simulate circular motion
    float r = 20.f;
    for (int i = 0; i < 8; ++i) {
        float theta = static_cast<float>(i) * 0.1f;
        predictor.update({makeObj(1, r*std::cos(theta), r*std::sin(theta))});
    }
    EXPECT_NO_THROW(predictor.predict(1));
}

TEST(TrajectoryPredictor, PredictAll_CoversAllObjects) {
    TrajectoryPredictor predictor;
    predictor.update({makeObj(1, 10, 0), makeObj(2, 20, 3)});
    predictor.update({makeObj(1, 11, 0), makeObj(2, 21, 3)});

    auto all = predictor.predictAll();
    EXPECT_EQ(all.size(), 2u);
    EXPECT_TRUE(all.count(1));
    EXPECT_TRUE(all.count(2));
}
