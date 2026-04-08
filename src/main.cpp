/// main.cpp – ADAS Perception Stack Demo
///
/// This program simulates a full ADAS processing pipeline using synthetic
/// sensor data. In a production vehicle all data feeds would be replaced by
/// real camera/radar/lidar drivers (or ROS 2 topic subscriptions).
///
///  Pipeline per frame:
///    CameraFrame ──► ObjectDetector ──┐
///    RadarFrame  ──────────────────■ SensorFusion ──► TrackedObjects
///    LidarFrame  ──────────────────┘         │
///                                            ▼
///    CameraFrame ──► LaneDetector ──► LaneInfo
///                                            │
///                                    TrajectoryPredictor
///                                            │
///                              ┌─────────────┼──────────────┐
///                          ACC │         AEB │          LKA │
///                              └─────────────┴──────────────┘
///                                         VehicleControl
///

#include "common/types.hpp"
#include "common/logger.hpp"
#include "common/math_utils.hpp"
#include "perception/object_detector.hpp"
#include "perception/lane_detector.hpp"
#include "perception/sensor_fusion.hpp"
#include "tracking/multi_object_tracker.hpp"
#include "prediction/trajectory_predictor.hpp"
#include "planning/adas_controllers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace adas;

// ─── Graceful shutdown ────────────────────────────────────────────────────────
static volatile bool gRunning = true;
static void sigHandler(int) { gRunning = false; }

// ─── Synthetic data generators ────────────────────────────────────────────────
namespace synthetic {

/// Generate a greyscale camera frame (noise pattern)
CameraFrame makeFrame(int w, int h, uint32_t seed) {
    CameraFrame f;
    f.width    = w;
    f.height   = h;
    f.channels = 3;
    f.data.resize(w * h * 3);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& px : f.data) px = static_cast<uint8_t>(dist(rng));
    f.stamp = now();
    return f;
}

/// Generate plausible radar targets around the ego vehicle
RadarFrame makeRadar(float egoSpeed, int nTargets, uint32_t seed) {
    RadarFrame rf;
    rf.stamp = now();
    std::mt19937 rng(seed);
    std::normal_distribution<float> distR(25.f, 8.f);
    std::uniform_real_distribution<float> distAz(-0.3f, 0.3f);
    std::normal_distribution<float> distRRate(-egoSpeed * 0.8f, 2.f);

    for (int i = 0; i < nTargets; ++i) {
        RadarTarget t;
        t.range     = std::abs(distR(rng));
        t.azimuth   = distAz(rng);
        t.rangeRate = distRRate(rng);
        t.rcs       = 10.f;
        rf.targets.push_back(t);
    }
    return rf;
}

/// Simulate a set of object detections from the "model output"
std::vector<DetectedObject> makeDetections(float egoSpeed, uint32_t seed) {
    std::vector<DetectedObject> dets;
    std::mt19937 rng(seed);

    // Lead car ~30 m ahead
    {
        DetectedObject d;
        d.id         = 1;
        d.classId    = ObjectClass::Car;
        d.confidence = 0.92f;
        d.bbox2d     = {250.f, 180.f, 140.f, 100.f};
        d.distance   = 30.f;
        d.velocity   = egoSpeed * 0.6f;   // slower than ego
        d.bbox3d.center.x =  30.f;
        d.bbox3d.center.y =   0.2f;
        d.stamp      = now();
        dets.push_back(d);
    }
    // Pedestrian on the right shoulder
    {
        DetectedObject d;
        d.id         = 2;
        d.classId    = ObjectClass::Pedestrian;
        d.confidence = 0.78f;
        d.bbox2d     = {520.f, 200.f, 50.f, 120.f};
        d.distance   = 15.f;
        d.velocity   = 1.2f;
        d.bbox3d.center.x =  15.f;
        d.bbox3d.center.y =  3.5f;
        d.stamp      = now();
        dets.push_back(d);
    }
    // Red traffic light 25 m away
    {
        DetectedObject d;
        d.id         = 3;
        d.classId    = ObjectClass::TrafficLight;
        d.confidence = 0.88f;
        d.bbox2d     = {300.f, 60.f, 40.f, 80.f};
        d.distance   = 25.f;
        d.tlColor    = TrafficLightColor::Green;  // chanced to Red scenario below
        d.bbox3d.center.x = 25.f;
        d.bbox3d.center.y =  0.f;
        d.stamp      = now();
        dets.push_back(d);
    }
    return dets;
}

/// Simulate ego state
EgoState makeEgo(float speed, int frame) {
    EgoState ego;
    ego.speed        = speed;
    ego.acceleration = 0.2f * std::sin(frame * 0.1f);
    ego.yawRate      = 0.01f * std::cos(frame * 0.05f);
    ego.stamp        = now();
    return ego;
}

/// Simulate a simple lane info (slightly departing on even frames)
LaneInfo makeLane(int frame) {
    LaneInfo info;
    LaneBoundary left, right;
    left.type  = LaneType::Solid;
    right.type = LaneType::Solid;
    for (int y = 200; y <= 480; y += 10) {
        left.points.push_back({180.f, static_cast<float>(y)});
        right.points.push_back({500.f, static_cast<float>(y)});
    }
    info.left          = left;
    info.right         = right;
    info.lateralError  = (frame % 60 < 30) ?  0.05f : 0.35f;  // occasional drift
    info.headingError  = 0.02f * std::sin(frame * 0.1f);
    info.isDeparting   = std::abs(info.lateralError) > 0.3f;
    return info;
}

} // namespace synthetic

// ─── Print VehicleControl ─────────────────────────────────────────────────────
static void printControl(const VehicleControl& c, const std::string& src) {
    LOG_INFO("  [" << src << "]"
             << "  thr=" << c.throttle
             << "  brk=" << c.brake
             << "  steer=" << c.steering
             << (c.aebActive ? " [AEB]" : "")
             << (c.lkaActive ? " [LKA]" : "")
             << (c.accActive ? " [ACC]" : ""));
}

// ─── Merge individual controller outputs ─────────────────────────────────────
static VehicleControl mergeControls(const VehicleControl& acc,
                                     const VehicleControl& aeb,
                                     const VehicleControl& lka,
                                     const VehicleControl& tl)
{
    VehicleControl out;
    // AEB and traffic-light braking takes priority
    out.brake     = std::max({acc.brake, aeb.brake, tl.brake});
    out.throttle  = (out.brake > 0.01f) ? 0.f : acc.throttle;
    out.steering  = lka.steering;           // lateral overlay
    out.aebActive = aeb.aebActive;
    out.lkaActive = lka.lkaActive;
    out.accActive = acc.accActive;
    return out;
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::signal(SIGINT,  sigHandler);
    std::signal(SIGTERM, sigHandler);

    // ── Logger setup ─────────────────────────────────────────────────────────
    Logger::instance().setLevel(LogLevel::Debug);
    Logger::instance().addFileSink("adas_demo.log");

    LOG_INFO("╔══════════════════════════════════════════════════╗");
    LOG_INFO("║       ADAS Perception Stack – Demo               ║");
    LOG_INFO("╚══════════════════════════════════════════════════╝");

    // ── Parse optional model paths from argv ─────────────────────────────────
    std::string detectorModel  = (argc > 1) ? argv[1] : "";
    std::string laneModel      = (argc > 2) ? argv[2] : "";

    // ── Instantiate pipeline components ──────────────────────────────────────
    perception::ObjectDetectorConfig detCfg;
    detCfg.modelPath            = detectorModel;
    detCfg.confidenceThreshold  = 0.45f;
    detCfg.nmsIouThreshold      = 0.45f;

    perception::LaneDetectorConfig laneCfg;
    laneCfg.modelPath = laneModel;

    perception::ObjectDetector detector(detCfg);
    perception::LaneDetector   laneDetector(laneCfg);
    perception::SensorFusion   fusion;

    tracking::MultiObjectTracker tracker;
    prediction::TrajectoryPredictor predictor;

    planning::ACCController  acc;
    planning::AEBController  aeb;
    planning::LaneKeepingAssist lka;
    planning::TrafficLightHandler tlHandler;

    // ── Set a 30 m/s (≈108 km/h) cruise target ───────────────────────────────
    acc.setSpeed(30.f);

    // ── Simulation state ─────────────────────────────────────────────────────
    float egoSpeed     = 25.f;  // m/s
    double dt          = 1.0 / 30.0;
    uint64_t frameNo   = 0;
    auto tStart        = std::chrono::steady_clock::now();

    LOG_INFO("Starting pipeline at 30 Hz (synthetic data)...");

    while (gRunning) {
        auto tFrame = std::chrono::steady_clock::now();
        ++frameNo;

        // ── Generate synthetic sensor inputs ─────────────────────────────────
        auto frame  = synthetic::makeFrame(640, 480, frameNo);
        auto radar  = synthetic::makeRadar(egoSpeed, 3, frameNo);
        auto ego    = synthetic::makeEgo(egoSpeed, frameNo);
        auto lane   = synthetic::makeLane(frameNo);
        auto rawDets = synthetic::makeDetections(egoSpeed, frameNo);

        // ── Perception ───────────────────────────────────────────────────────
        // If a real model is loaded, run inference; otherwise use synthetic dets
        std::vector<DetectedObject> detections;
        if (detector.isReady()) {
            detections = detector.detect(frame);
        } else {
            detections = rawDets;
        }

        // ── Sensor Fusion ────────────────────────────────────────────────────
        fusion.update(detections, radar, nullptr, ego);
        auto fusedObjects = fusion.getTrackedObjects();

        // If fusion hasn't warmed up yet, fall back to raw detections
        if (fusedObjects.empty()) fusedObjects = detections;

        // ── MOT (image-space tracking for IDs) ───────────────────────────────
        auto tracks = tracker.update(detections);

        // ── Trajectory prediction ────────────────────────────────────────────
        predictor.update(fusedObjects);
        auto trajectories = predictor.predictAll();

        // ── Planning ─────────────────────────────────────────────────────────
        // Find the nearest in-path lead vehicle for ACC
        std::optional<DetectedObject> lead;
        float minDist = 120.f;
        for (const auto& obj : fusedObjects) {
            if (obj.classId == ObjectClass::Car
             || obj.classId == ObjectClass::Truck) {
                if (obj.distance < minDist
                 && std::abs(obj.bbox3d.center.y) < 2.0f) {
                    minDist = obj.distance;
                    lead    = obj;
                }
            }
        }

        auto accCtrl = acc.update(ego, lead, dt);
        auto aebCtrl = aeb.update(fusedObjects, ego);
        auto lkaCtrl = lka.update(lane, ego, dt);
        auto tlCtrl  = tlHandler.update(fusedObjects, ego, dt);

        auto finalCtrl = mergeControls(accCtrl, aebCtrl, lkaCtrl, tlCtrl);

        // ── Simple ego-speed integration (toy model) ─────────────────────────
        float netAccel = finalCtrl.throttle * 2.0f - finalCtrl.brake * 5.0f;
        egoSpeed = math::clamp(egoSpeed + netAccel * static_cast<float>(dt),
                               0.f, 50.f);

        // ── Periodic status log ───────────────────────────────────────────────
        if (frameNo % 30 == 0) {
            double elapsed = std::chrono::duration<double>(
                tFrame - tStart).count();
            LOG_INFO("── Frame " << frameNo
                     << "  t=" << static_cast<int>(elapsed) << "s"
                     << "  ego=" << egoSpeed << " m/s"
                     << "  objects=" << fusedObjects.size()
                     << "  tracks=" << tracks.size()
                     << "  AEB=" << planning::toString(aeb.currentState())
                     << "  TL=" << static_cast<int>(tlHandler.currentPhase())
                     << "  lane_dep=" << (lane.isDeparting ? "YES" : "no") );
            printControl(finalCtrl, "merged");
        }

        // ── Print trajectory for lead object every 5 seconds ─────────────────
        if (frameNo % 150 == 0 && lead.has_value()) {
            auto traj = predictor.predict(lead->id);
            if (!traj.empty()) {
                LOG_INFO("  Lead-vehicle trajectory (" << traj.size()
                         << " pts, horizon 3s):");
                for (size_t i = 0; i < std::min(traj.size(), size_t{5}); ++i) {
                    LOG_INFO("    t+" << traj[i].timeOffset << "s -> ("
                             << traj[i].position.x << ", "
                             << traj[i].position.y << ") m");
                }
            }
        }

        // ── 30 Hz timing ──────────────────────────────────────────────────────
        auto tEnd    = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(tEnd - tFrame).count();
        double sleep = dt - elapsed;
        if (sleep > 0)
            std::this_thread::sleep_for(
                std::chrono::duration<double>(sleep));
    }

    double totalTime = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - tStart).count();
    LOG_INFO("Demo finished. Ran " << frameNo << " frames in "
             << totalTime << " s  ("
             << frameNo / totalTime << " fps avg)");
    return 0;
}
