# 🚗 ADAS Functions Reference

Detailed description of the four ADAS planning functions implemented in the stack: **ACC**, **AEB**, **LKA**, and **Traffic-Light Handler**.

---

## Table of Contents

1. [Adaptive Cruise Control (ACC)](#1-adaptive-cruise-control-acc)
2. [Automatic Emergency Braking (AEB)](#2-automatic-emergency-braking-aeb)
3. [Lane Keeping Assist (LKA)](#3-lane-keeping-assist-lka)
4. [Traffic-Light Handler](#4-traffic-light-handler)
5. [Combined Output Merging](#5-combined-output-merging)

---

## 1. Adaptive Cruise Control (ACC)

### Purpose

Maintain a **set speed** while keeping a safe **time headway** to the closest in-lane lead vehicle.

### Configuration (`ACCControllerConfig`)

| Parameter | Default | Description |
|---|---|---|
| `setSpeed` | 30.0 m/s | Target cruise speed |
| `timeHeadway` | 2.0 s | Desired gap in seconds |
| `minGap` | 5.0 m | Minimum gap regardless of speed |
| `maxAcceleration` | 2.0 m/s² | Throttle authority |
| `maxDeceleration` | 3.5 m/s² | Brake authority |
| `kp`, `ki`, `kd` | 0.4, 0.05, 0.1 | PID gains |

### Headway Gap Formula

$$d_{\text{desired}} = d_{\text{min}} + \tau \cdot v_{\text{ego}}$$

where $\tau$ is `timeHeadway` and $v_{\text{ego}}$ is current ego speed in m/s.

### Control Logic

```
if no lead vehicle within sensor range:
    error = setSpeed − egoSpeed
    throttle = PID.update(error, dt)
    brake = 0.0

else:
    d_desired = minGap + timeHeadway × egoSpeed
    gap_error = distance_to_lead − d_desired

    if gap_error > 0 and egoSpeed < setSpeed:
        # Free space and below set speed — accelerate
        speed_error = min(setSpeed, lead_speed) − egoSpeed
        throttle = PID.update(speed_error, dt)
        brake = 0.0

    elif gap_error < 0:
        # Too close — brake proportionally
        throttle = 0.0
        brake = clamp(|gap_error| / d_desired × maxDeceleration, 0, 1)
```

### PID Tuning Notes

- `Kp = 0.4` — moderate proportional response, avoids oscillation at highway speed
- `Ki = 0.05` — removes steady-state speed error (e.g., slight uphill grade)
- `Kd = 0.10` — damps oscillation when approaching lead vehicle
- Anti-windup clamp: `±10.0` throttle-equivalent units

---

## 2. Automatic Emergency Braking (AEB)

### Purpose

Detect an **imminent collision** with any object in the ego vehicle's path and automatically apply braking to avoid or mitigate the crash.

### Configuration (`AEBControllerConfig`)

| Parameter | Default | Description |
|---|---|---|
| `ttcWarning` | 3.5 s | TTC threshold for Warning state |
| `ttcPartialBrake` | 2.5 s | TTC threshold for Partial Brake |
| `ttcFullBrake` | 1.5 s | TTC threshold for Full Brake |
| `minConfidence` | 0.4 | Minimum detection confidence to consider |
| `partialBrakeForce` | 0.4 | Brake pedal value [0, 1] |
| `fullBrakeForce` | 1.0 | Maximum brake pedal force |
| `lateralGateHalfWidth` | 1.8 m | Half-width of path corridor |

### Time-To-Collision

$$\text{TTC} = \frac{d_{\text{range}}}{v_{\text{closing}}}$$

where $v_{\text{closing}} = v_{\text{ego}} - v_{\text{lead}}$ (positive when closing).

Only computed when $v_{\text{closing}} > 0$. Objects with $v_{\text{closing}} \leq 0$ (moving away or stationary relative) are ignored.

### In-Path Check

An object is considered **in the ego vehicle's path** when its lateral position satisfies:

$$|y_{\text{object}} - y_{\text{ego}}| \leq \text{lateralGateHalfWidth}$$

The check is applied in ego-vehicle coordinates.

### AEB State Machine

```
                     TTC > 3.5s
           ┌──────────────────────────────┐
           │                              │
           ▼                              │
      ┌─────────┐  TTC ≤ 3.5s  ┌──────────────┐
      │INACTIVE │────────────►  │   WARNING    │  ◄── visual/audio alert
      └─────────┘               └──────────────┘
                                       │ TTC ≤ 2.5s
                                       ▼
                               ┌──────────────────┐
                               │  PARTIAL_BRAKE   │  brake = 0.40
                               └──────────────────┘
                                       │ TTC ≤ 1.5s
                                       ▼
                               ┌──────────────────┐
                               │   FULL_BRAKE     │  brake = 1.00
                               └──────────────────┘
```

Transitions go **downward only** within a single frame. State resets to `INACTIVE` when the minimum TTC across all in-path objects exceeds `ttcWarning`.

### Multi-Object Handling

All tracked objects are evaluated. The **minimum TTC** among in-path objects with sufficient confidence drives the state transition. This ensures the most critical threat governs the response.

---

## 3. Lane Keeping Assist (LKA)

### Purpose

Apply a small corrective **steering torque** to keep the vehicle centred in the detected lane when the driver's steering input is insufficient.

### Configuration (`LKAConfig`)

| Parameter | Default | Description |
|---|---|---|
| `minSpeed` | 10.0 m/s | Below this speed, LKA is inactive |
| `maxSteering` | 0.3 (rad) | Output steering clamped to ±this |
| `lateralKp` | 0.08 | Proportional gain on lateral error |
| `lateralKd` | 0.02 | Derivative gain on lateral error |
| `headingKp` | 0.6 | Proportional gain on heading error |
| `headingKd` | 0.1 | Derivative gain on heading error |

### Control Law

The steering command is the sum of two PD terms:

$$\delta = K_{p,\text{lat}} \cdot e_{\text{lat}} + K_{d,\text{lat}} \cdot \dot{e}_{\text{lat}} + K_{p,\psi} \cdot e_{\psi} + K_{d,\psi} \cdot \dot{e}_{\psi}$$

where:

- $e_{\text{lat}}$ = lateral error (metres, positive = vehicle is to the right of lane centre)
- $\dot{e}_{\text{lat}}$ = derivative of lateral error between frames
- $e_{\psi}$ = heading error (radians, positive = vehicle heading right relative to lane)
- $\dot{e}_{\psi}$ = derivative of heading error

The output is then clamped: $\delta \in [-\delta_{\max}, +\delta_{\max}]$.

### Lane Error Computation (`LaneDetector::computeErrors`)

```
left_lane_x  = x-position of left lane boundary at ego row
right_lane_x = x-position of right lane boundary at ego row
lane_centre  = (left_lane_x + right_lane_x) / 2.0

lateral_error = image_centre_x − lane_centre    [pixels → metres via scale factor]
heading_error = atan(lane_midline_slope)         [radians]
```

### Activation Logic

LKA is engaged when:
1. `enabled == true` in config
2. `egoSpeed >= minSpeed` — prevents activation at low speed / parking
3. Valid `LaneInfo` received (at least one lane boundary detected)

When LKA is inactive, `steering = 0.0` and `lkaActive = false` in the output.

---

## 4. Traffic-Light Handler

### Purpose

Detect upcoming **red or amber traffic lights** and bring the vehicle smoothly to a stop at the stop line.

### Configuration (`TrafficLightConfig`)

| Parameter | Default | Description |
|---|---|---|
| `lookaheadDistance` | 30.0 m | Maximum stop-line detection range |
| `stopLineTolerance` | 1.0 m | Accepted stop precision (creep suppression) |
| `kp`, `ki`, `kd` | 0.4, 0.02, 0.05 | PID gains for stop-line distance |

### Control Logic

```
if traffic_light_color == GREEN or NO_LIGHT:
    pass — no intervention

elif traffic_light_color is RED or AMBER:
    if stop_line_distance <= 0.0 + stopLineTolerance:
        # At stop line — hold still
        brake = 1.0; throttle = 0.0

    elif stop_line_distance <= lookaheadDistance:
        # Approaching — PID on distance-to-stop
        brake = clamp(PID.update(stop_line_distance, dt), 0, 1)
        throttle = 0.0

    else:
        # Too far — no intervention
        pass
```

### Braking Profile

The PID is tuned to produce a **comfortable deceleration ramp**: near-zero brake at 30 m, reaching full brake within 5 m. This avoids harsh sudden stops while ensuring the vehicle halts before the stop line.

---

## 5. Combined Output Merging

The four controllers run independently and their outputs are merged with a **safety-priority hierarchy**:

```
VehicleControl merged_output;

// Steering: LKA takes authority if active
merged_output.steering = lka.isActive() ? lka_output.steering : driver_input.steering;

// Brake: AEB takes highest priority; then TL handler; then ACC
merged_output.brake = std::max({aeb_output.brake,
                                tl_output.brake,
                                acc_output.brake});

// Throttle: zeroed if any brake > 0 threshold (prevents simultaneous throttle+brake)
merged_output.throttle = (merged_output.brake > 0.05f) ? 0.0f : acc_output.throttle;

// Status flags
merged_output.aebActive  = aeb_output.aebActive;
merged_output.lkaActive  = lka_output.lkaActive;
merged_output.accActive  = acc_output.accActive;
```

This means:
- **AEB always wins** on braking — it cannot be overridden by ACC or TL handler.
- **LKA** only steers when its PD output is non-trivial; does not fight driver.
- **ACC** throttle is cut whenever any safety system requests brake > 5%.
