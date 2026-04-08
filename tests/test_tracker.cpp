/// test_tracker.cpp
/// Unit tests for the SORT-based Multi-Object Tracker.
#include <gtest/gtest.h>
#include "tracking/multi_object_tracker.hpp"

using namespace adas;
using namespace adas::tracking;

static DetectedObject makeDet(float cx, float cy, float w, float h,
                               ObjectClass cls = ObjectClass::Car,
                               float conf = 0.9f)
{
    DetectedObject o;
    o.classId    = cls;
    o.confidence = conf;
    o.bbox2d     = {cx - w/2.f, cy - h/2.f, w, h};
    o.stamp      = now();
    return o;
}

// ─── Tests ───────────────────────────────────────────────────────────────────
TEST(MultiObjectTracker, InitialState_Empty) {
    MultiObjectTracker mot;
    EXPECT_EQ(mot.trackCount(), 0u);
}

TEST(MultiObjectTracker, SingleDetection_CreatesTrack) {
    MOTConfig cfg;
    cfg.minHits = 1;
    MultiObjectTracker mot(cfg);

    auto tracks = mot.update({makeDet(320.f, 240.f, 100.f, 80.f)});
    EXPECT_EQ(tracks.size(), 1u);
}

TEST(MultiObjectTracker, NewDetection_AssignedSameID_OverFrames) {
    MOTConfig cfg;
    cfg.minHits      = 3;
    cfg.iouThreshold = 0.2f;
    MultiObjectTracker mot(cfg);

    auto det = makeDet(320.f, 240.f, 100.f, 80.f);
    mot.update({det});
    mot.update({det});
    auto tracks = mot.update({det});

    ASSERT_EQ(tracks.size(), 1u);
    uint64_t id1 = tracks[0].id;

    // 4th frame – same position
    auto tracks2 = mot.update({det});
    ASSERT_EQ(tracks2.size(), 1u);
    EXPECT_EQ(tracks2[0].id, id1);  // ID must persist
}

TEST(MultiObjectTracker, LostTrack_Removed_AfterMaxAge) {
    MOTConfig cfg;
    cfg.minHits = 1;
    cfg.maxAge  = 3;
    MultiObjectTracker mot(cfg);

    mot.update({makeDet(320.f, 240.f, 100.f, 80.f)});
    // Update with no detections – track should age out
    for (int i = 0; i < 5; ++i) mot.update({});

    EXPECT_EQ(mot.trackCount(), 0u);
}

TEST(MultiObjectTracker, TwoObjects_GetDistinctIDs) {
    MOTConfig cfg;
    cfg.minHits = 1;
    MultiObjectTracker mot(cfg);

    auto d1 = makeDet(100.f, 240.f, 60.f, 50.f);
    auto d2 = makeDet(400.f, 240.f, 60.f, 50.f);

    auto tracks = mot.update({d1, d2});
    ASSERT_EQ(tracks.size(), 2u);
    EXPECT_NE(tracks[0].id, tracks[1].id);
}

TEST(MultiObjectTracker, MergedNearbyBoxes_NotConfused) {
    MOTConfig cfg;
    cfg.minHits = 1;
    cfg.iouThreshold = 0.3f;
    MultiObjectTracker mot(cfg);

    // Two well-separated objects
    auto d1 = makeDet(100.f, 100.f, 50.f, 50.f);
    auto d2 = makeDet(400.f, 100.f, 50.f, 50.f);

    for (int i = 0; i < 3; ++i) mot.update({d1, d2});
    EXPECT_EQ(mot.trackCount(), 2u);
}

TEST(MultiObjectTracker, Reset_ClearsAllTracks) {
    MOTConfig cfg;
    cfg.minHits = 1;
    MultiObjectTracker mot(cfg);
    mot.update({makeDet(320.f, 240.f, 100.f, 80.f)});
    ASSERT_GT(mot.trackCount(), 0u);

    mot.reset();
    EXPECT_EQ(mot.trackCount(), 0u);
}
