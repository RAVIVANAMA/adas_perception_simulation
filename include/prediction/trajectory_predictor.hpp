#pragma once

#include "common/types.hpp"
#include <vector>
#include <unordered_map>

namespace adas::prediction {

enum class PredictionModel {
    ConstantVelocity,     // simple CV linear extrapolation
    ConstantTurnRate,     // CTRA (constant turn-rate & acceleration)
    PolynomialFit         // cubic spline fit to recent history
};

struct PredictorConfig {
    PredictionModel model{PredictionModel::ConstantTurnRate};
    float           horizonSeconds{3.0f};    // how far ahead to predict
    float           dtSeconds{0.1f};         // time step between trajectory points
    size_t          historyFrames{10};        // frames of history kept per object
};

/// Trajectory predictor – given a set of tracked objects (with history),
/// returns a predicted trajectory for each object over the configured horizon.
class TrajectoryPredictor {
public:
    explicit TrajectoryPredictor(PredictorConfig cfg = {});

    /// Feed the current frame's tracked objects every cycle.
    void update(const std::vector<DetectedObject>& objects);

    /// Retrieve the predicted trajectory for a given object ID.
    /// Returns empty vector if the ID is unknown.
    Trajectory predict(uint64_t objectId) const;

    /// Predict for all currently tracked objects.
    std::unordered_map<uint64_t, Trajectory> predictAll() const;

    void reset();

private:
    using History = std::vector<DetectedObject>;

    Trajectory cvPredict(const History& h) const;
    Trajectory ctraPredict(const History& h) const;
    Trajectory polyPredict(const History& h) const;

    PredictorConfig                               cfg_;
    std::unordered_map<uint64_t, History>         histories_;
};

} // namespace adas::prediction
