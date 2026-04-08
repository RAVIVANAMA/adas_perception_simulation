/// test_math_utils.cpp
/// Unit tests for common math utilities: PID, NMS, TTC, IoU, KalmanFilter1D.
#include <gtest/gtest.h>
#include "common/math_utils.hpp"
#include "common/types.hpp"
#include <cmath>

using namespace adas;
using namespace adas::math;

// ─── PID controller ───────────────────────────────────────────────────────────
TEST(PID, ZeroError_ZeroOutput) {
    PID<double> pid(1.0, 0.0, 0.0, -10.0, 10.0);
    EXPECT_NEAR(pid.update(5.0, 5.0, 0.1), 0.0, 1e-9);
}

TEST(PID, PositiveError_PositiveOutput) {
    PID<double> pid(1.0, 0.0, 0.0, -100.0, 100.0);
    double out = pid.update(10.0, 0.0, 0.1);
    EXPECT_GT(out, 0.0);
}

TEST(PID, Output_Clamped) {
    PID<double> pid(100.0, 0.0, 0.0, -5.0, 5.0);  // high gain
    double out = pid.update(1000.0, 0.0, 0.1);
    EXPECT_LE(out,  5.0);
    EXPECT_GE(out, -5.0);
}

TEST(PID, Reset_ClearsIntegral) {
    PID<double> pid(0.0, 1.0, 0.0, -100.0, 100.0);  // I only
    for (int i = 0; i < 10; ++i) pid.update(1.0, 0.0, 0.1);
    pid.reset();
    double out = pid.update(0.0, 0.0, 0.1);
    EXPECT_NEAR(out, 0.0, 1e-9);
}

// ─── TTC ─────────────────────────────────────────────────────────────────────
TEST(MathUtils, TTC_Basic) {
    float ttc = computeTTC(30.f, 10.f);
    EXPECT_NEAR(ttc, 3.f, 1e-5f);
}

TEST(MathUtils, TTC_Diverging_IsNaN) {
    float ttc = computeTTC(30.f, -2.f);
    EXPECT_TRUE(std::isnan(ttc));
}

TEST(MathUtils, TTC_ZeroClosingSpeed_IsNaN) {
    EXPECT_TRUE(std::isnan(computeTTC(30.f, 0.f)));
}

// ─── IoU ──────────────────────────────────────────────────────────────────────
TEST(MathUtils, IoU_IdenticalBoxes) {
    AABB a{0,0,100,100};
    EXPECT_FLOAT_EQ(iou(a, a), 1.f);
}

TEST(MathUtils, IoU_NoOverlap) {
    AABB a{0,0,10,10}, b{20,20,10,10};
    EXPECT_FLOAT_EQ(iou(a, b), 0.f);
}

TEST(MathUtils, IoU_HalfOverlap) {
    AABB a{0,0,10,10}, b{5,0,10,10};
    // intersection = 5×10=50, union = 150
    EXPECT_NEAR(iou(a, b), 50.f/150.f, 1e-5f);
}

// ─── Kalman1D ─────────────────────────────────────────────────────────────────
TEST(KalmanFilter1D, ConvergesOnConstantSignal) {
    KalmanFilter1D kf(0.01, 0.1, 0.0);
    for (int i = 0; i < 50; ++i) kf.update(5.0);
    EXPECT_NEAR(kf.value(), 5.0, 0.1);
}

TEST(KalmanFilter1D, Tracks_StepChange) {
    KalmanFilter1D kf(0.1, 0.5, 0.0);
    for (int i = 0; i < 20; ++i) kf.update(0.0);
    for (int i = 0; i < 30; ++i) kf.update(10.0);
    EXPECT_NEAR(kf.value(), 10.0, 0.5);
}

// ─── Angle wrap ───────────────────────────────────────────────────────────────
TEST(MathUtils, WrapAngle_LargePositive) {
    double a = wrapAngle(5.0);  // 5 - 2π ≈ -1.28
    EXPECT_LT(a,  PI);
    EXPECT_GE(a, -PI);
}

TEST(MathUtils, WrapAngle_NegativeLarge) {
    double a = wrapAngle(-10.0);
    EXPECT_LT(a,  PI);
    EXPECT_GE(a, -PI);
}

// ─── NMS ──────────────────────────────────────────────────────────────────────
TEST(MathUtils, NMS_RemovesHighOverlapBoxes) {
    using Box = BoundingBox2D;
    std::vector<Box>   boxes  = {{0,0,100,100},{5,5,100,100},{200,200,50,50}};
    std::vector<float> scores = {0.9f, 0.85f, 0.6f};

    auto kept = nms(boxes, scores, 0.5f, 0.0f);

    // Box 0 and 1 heavily overlap → only box 0 (higher score) kept, plus box 2
    EXPECT_EQ(kept.size(), 2u);
    EXPECT_NE(std::find(kept.begin(), kept.end(), 0u), kept.end());
    EXPECT_NE(std::find(kept.begin(), kept.end(), 2u), kept.end());
}

TEST(MathUtils, NMS_ScoreThreshold_FiltersLow) {
    using Box = BoundingBox2D;
    std::vector<Box>   boxes  = {{0,0,10,10},{100,100,10,10}};
    std::vector<float> scores = {0.9f, 0.2f};

    auto kept = nms(boxes, scores, 0.5f, 0.5f);  // threshold = 0.5

    EXPECT_EQ(kept.size(), 1u);
    EXPECT_EQ(kept[0], 0u);
}
