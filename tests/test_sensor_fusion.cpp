/// test_sensor_fusion.cpp
/// Unit tests for SensorFusion (EKF-based multi-sensor track management).
#include <gtest/gtest.h>
#include "perception/sensor_fusion.hpp"

using namespace adas;
using namespace adas::perception;

static DetectedObject makeCamDet(uint64_t id, float dist, float lat,
                                  ObjectClass cls = ObjectClass::Car,
                                  float conf = 0.9f)
{
    DetectedObject o;
    o.id              = id;
    o.classId         = cls;
    o.confidence      = conf;
    o.distance        = dist;
    o.bbox2d          = {300.f, 200.f, 100.f, 80.f};
    o.bbox3d.center.x = dist;
    o.bbox3d.center.y = lat;
    o.stamp           = now();
    return o;
}

static EgoState makeEgo(float speed = 20.f) {
    EgoState e; e.speed = speed; e.stamp = now(); return e;
}

// ─── Tests ───────────────────────────────────────────────────────────────────
TEST(SensorFusion, InitialState_Empty) {
    SensorFusion fusion;
    EXPECT_TRUE(fusion.tracks().empty());
}

TEST(SensorFusion, TrackCreated_AfterDetection) {
    SensorFusion fusion;
    RadarFrame radar;
    auto ego = makeEgo();

    auto det = makeCamDet(1, 30.f, 0.f);
    fusion.update({det}, radar, nullptr, ego);

    // Track created but not yet confirmed (minHits = 3)
    EXPECT_EQ(fusion.tracks().size(), 1u);
}

TEST(SensorFusion, TrackConfirmed_AfterMinHits) {
    SensorFusionConfig cfg;
    cfg.minHitsToConfirm = 2;
    SensorFusion fusion(cfg);
    RadarFrame radar;
    auto ego = makeEgo();

    auto det = makeCamDet(1, 30.f, 0.f);
    for (int i = 0; i < 3; ++i)
        fusion.update({det}, radar, nullptr, ego);

    auto objects = fusion.getTrackedObjects();
    EXPECT_FALSE(objects.empty());
}

TEST(SensorFusion, StaleTrack_Removed) {
    SensorFusionConfig cfg;
    cfg.maxLostFrames    = 3;
    cfg.minHitsToConfirm = 1;
    SensorFusion fusion(cfg);
    RadarFrame radar;
    auto ego = makeEgo();

    // Insert once, then update with no detections to age out the track
    fusion.update({makeCamDet(1, 30.f, 0.f)}, radar, nullptr, ego);
    for (int i = 0; i < 5; ++i)
        fusion.update({}, radar, nullptr, ego);

    EXPECT_TRUE(fusion.tracks().empty());
}

TEST(SensorFusion, MultipleObjects_TrackedIndependently) {
    SensorFusion fusion;
    RadarFrame radar;
    auto ego = makeEgo();

    auto d1 = makeCamDet(1, 30.f,  0.f, ObjectClass::Car);
    auto d2 = makeCamDet(2, 50.f,  3.f, ObjectClass::Truck);
    fusion.update({d1, d2}, radar, nullptr, ego);

    EXPECT_EQ(fusion.tracks().size(), 2u);
}

TEST(SensorFusion, RadarFusion_DoesNotCrash) {
    SensorFusion fusion;
    auto ego = makeEgo();

    RadarFrame radar;
    RadarTarget tgt;
    tgt.range     = 28.f;
    tgt.azimuth   = 0.05f;
    tgt.rangeRate = -5.f;
    radar.targets.push_back(tgt);

    auto det = makeCamDet(1, 28.f, 0.f);
    // Should not throw
    EXPECT_NO_THROW(fusion.update({det}, radar, nullptr, ego));
}

TEST(SensorFusion, Reset_ClearsAllTracks) {
    SensorFusion fusion;
    RadarFrame radar;
    auto ego = makeEgo();
    fusion.update({makeCamDet(1, 30.f, 0.f)}, radar, nullptr, ego);
    ASSERT_FALSE(fusion.tracks().empty());

    fusion.reset();
    EXPECT_TRUE(fusion.tracks().empty());
}
