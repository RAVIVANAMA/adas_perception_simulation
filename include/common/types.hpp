#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <optional>
#include <chrono>
#include <Eigen/Dense>   // Eigen is header-only; include path set via CMake

namespace adas {

// ─── Timestamp ───────────────────────────────────────────────────────────────
using Timestamp = std::chrono::time_point<std::chrono::steady_clock>;

inline Timestamp now() { return std::chrono::steady_clock::now(); }

inline double toSeconds(Timestamp t) {
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

// ─── 2-D / 3-D geometry primitives ───────────────────────────────────────────
struct Point2D { float x{0.f}, y{0.f}; };
struct Point3D { float x{0.f}, y{0.f}, z{0.f}; };

struct BoundingBox2D {
    float x{0.f}, y{0.f};   // top-left corner (pixels)
    float width{0.f}, height{0.f};

    float area()   const { return width * height; }
    float cx()     const { return x + width  / 2.f; }
    float cy()     const { return y + height / 2.f; }
    float right()  const { return x + width; }
    float bottom() const { return y + height; }

    // Intersection-over-Union with another box
    float iou(const BoundingBox2D& o) const noexcept {
        float ix1 = std::max(x, o.x);
        float iy1 = std::max(y, o.y);
        float ix2 = std::min(x + width,  o.x + o.width);
        float iy2 = std::min(y + height, o.y + o.height);
        if (ix2 <= ix1 || iy2 <= iy1) return 0.f;
        float inter = (ix2 - ix1) * (iy2 - iy1);
        float uni   = area() + o.area() - inter;
        return (uni > 0.f) ? inter / uni : 0.f;
    }
};

struct BoundingBox3D {
    Point3D center;
    float   length{0.f}, width{0.f}, height{0.f};
    float   yaw{0.f};   // heading in radians
};

// ─── Object class labels ─────────────────────────────────────────────────────
enum class ObjectClass : uint8_t {
    Unknown      = 0,
    Car          = 1,
    Truck        = 2,
    Pedestrian   = 3,
    Cyclist      = 4,
    Motorcycle   = 5,
    TrafficLight = 6,
    TrafficSign  = 7,
    Animal       = 8,
    COUNT
};

inline std::string toString(ObjectClass c) {
    static constexpr std::array<const char*, 9> names{
        "Unknown","Car","Truck","Pedestrian","Cyclist",
        "Motorcycle","TrafficLight","TrafficSign","Animal"
    };
    return names[static_cast<uint8_t>(c)];
}

// ─── Traffic-light state ──────────────────────────────────────────────────────
enum class TrafficLightColor : uint8_t { Unknown=0, Red, Amber, Green };

// ─── Detected object (output of object detector) ─────────────────────────────
struct DetectedObject {
    uint64_t      id{0};
    ObjectClass   classId{ObjectClass::Unknown};
    float         confidence{0.f};
    BoundingBox2D bbox2d;              // image-space bounding box
    BoundingBox3D bbox3d;              // world-space (if available)
    float         velocity{0.f};      // m/s  (from radar or sensor fusion)
    float         distance{0.f};      // m  (to host vehicle)
    Timestamp     stamp;

    // Traffic-light specific
    TrafficLightColor tlColor{TrafficLightColor::Unknown};
};

// ─── Lane information ─────────────────────────────────────────────────────────
enum class LaneType : uint8_t { Unknown=0, Solid, Dashed, DoubleSolid };

struct LaneBoundary {
    std::vector<Point2D> points;    // image-space polyline
    LaneType type{LaneType::Unknown};
    float    lateralOffset{0.f};    // m, signed (positive = left)
};

struct LaneInfo {
    std::optional<LaneBoundary> left;
    std::optional<LaneBoundary> right;
    float  headingError{0.f};        // rad, vehicle heading vs lane heading
    float  lateralError{0.f};        // m, signed lateral deviation from center
    bool   isDeparting{false};
};

// ─── Sensor data frames ───────────────────────────────────────────────────────
struct CameraFrame {
    std::vector<uint8_t> data;
    int   width{0}, height{0}, channels{3};
    Timestamp stamp;
};

struct RadarTarget {
    float range{0.f};           // m
    float azimuth{0.f};         // rad
    float elevation{0.f};       // rad
    float rangeRate{0.f};       // m/s (doppler)
    float rcs{0.f};             // dBm²
    bool  isStationary{false};
};

struct LidarPoint {
    float x{0.f}, y{0.f}, z{0.f};
    float intensity{0.f};
    uint8_t ring{0};
};

using PointCloud = std::vector<LidarPoint>;

struct RadarFrame  { std::vector<RadarTarget> targets; Timestamp stamp; };
struct LidarFrame  { PointCloud               cloud;   Timestamp stamp; };

// ─── Ego-vehicle state ────────────────────────────────────────────────────────
struct EgoState {
    float speed{0.f};            // m/s
    float acceleration{0.f};     // m/s²
    float yawRate{0.f};          // rad/s
    float steeringAngle{0.f};    // rad
    Timestamp stamp;
};

// ─── Vehicle control output ───────────────────────────────────────────────────
struct VehicleControl {
    float throttle{0.f};   // [0, 1]
    float brake{0.f};      // [0, 1]
    float steering{0.f};   // rad, positive = left
    bool  aebActive{false};
    bool  lkaActive{false};
    bool  accActive{false};
};

// ─── Trajectory ───────────────────────────────────────────────────────────────
struct TrajectoryPoint {
    Point3D  position;
    float    speed{0.f};
    float    heading{0.f};
    double   timeOffset{0.0};   // seconds from now
};

using Trajectory = std::vector<TrajectoryPoint>;

} // namespace adas
