# State Space and Discretization

This document provides a comprehensive explanation of the state space components and discretization strategy used in this RL-based drone landing system.

## Overview

This project uses **tabular Q-learning**, which requires discrete states. The environment provides continuous observations from telemetry and vision sensors, which are normalized and discretized into bins to create a finite state space.

## State Space Components

The state consists of 10 features derived from drone telemetry and visual detection:

### 1. Attitude (Orientation)
- **pitch**: Pitch angle in radians
  - Range: -0.5 to +0.5 radians (~±28.6 degrees)
  - **Discretization**: 3 bins (low/neutral/high)
  - Physical meaning: Forward/backward tilt
  
- **roll**: Roll angle in radians
  - Range: -0.5 to +0.5 radians (~±28.6 degrees)
  - **Discretization**: 3 bins (low/neutral/high)
  - Physical meaning: Left/right tilt

### 2. Body-Frame Velocity
- **vx**: Velocity in body-frame x-axis (m/s)
  - Range: -0.8 to +0.8 m/s
  - **Discretization**: 3 bins (backward/stationary/forward)
  - Physical meaning: Forward/backward speed relative to drone orientation
  
- **vy**: Velocity in body-frame y-axis (m/s)
  - Range: -0.8 to +0.8 m/s
  - **Discretization**: 3 bins (left/stationary/right)
  - Physical meaning: Left/right speed relative to drone orientation

### 3. Altitude
- **altitude**: Height above ground (m)
  - Range: 0.2 to 2.3 meters
  - **Discretization**: 4 bins (very-low/low/medium/high)
  - Physical meaning: Vertical distance from landing surface
  - Note: Episodes terminate when altitude ≤ 0.31m

### 4. Visual Detection (YOLO-based)
- **cx**: Horizontal center of detected landing pad
  - Range: 0.0 to 1.0 (normalized image coordinates)
  - **Discretization**: 6 bins
  - Physical meaning: Left-to-right position of platform in camera view
  - Note: 0.5 = center, <0.5 = left, >0.5 = right

- **cy**: Vertical center of detected landing pad
  - Range: 0.0 to 1.0 (normalized image coordinates)
  - **Discretization**: 6 bins
  - Physical meaning: Top-to-bottom position of platform in camera view
  - Note: 0.5 = center, <0.5 = top, >0.5 = bottom

- **conf**: Detection confidence score
  - Range: 0.0 to 1.0
  - **Discretization**: 2 bins (low/high confidence)
  - Threshold: < 0.3 → bin 0 (low), ≥ 0.3 → bin 1 (high)
  - Physical meaning: How confident YOLO is about detecting the landing pad

### 5. Detection Delta (Motion Tracking)
- **dx**: Change in horizontal center position
  - Range: -0.5 to +0.5 (normalized)
  - **Discretization**: 3 bins (moving-left/stable/moving-right)
  - Physical meaning: How fast the platform appears to move horizontally in view

- **dy**: Change in vertical center position
  - Range: -0.5 to +0.5 (normalized)
  - **Discretization**: 3 bins (moving-up/stable/moving-down)
  - Physical meaning: How fast the platform appears to move vertically in view

## Discretization Process

The discretization happens in three stages (see `discretizer.py`):

### Stage 1: Normalization
Continuous features are normalized to [0, 1] range:
```
normalized_value = (clipped_value - min_range) / (max_range - min_range)
```
where `clipped_value = clip(raw_value, min_range, max_range)`

### Stage 2: Binning
Normalized values are mapped to discrete bins:

**For most features:**
```
bin_index = int(clip(normalized_value * n_bins, 0, n_bins - 1))
```

**For confidence (special case):**
- If normalized_value < 0.3: bin = 0
- If normalized_value ≥ 0.3: bin = 1

### Stage 3: State Encoding
Discretized features are encoded into a string key:
```
# Format: "pitch_roll_vx_vy_altitude_cx_cy_conf_dx_dy"
# where each value is a bin index (integer)
Example state_key: "1_1_1_2_2_3_3_1_1_1"
```

This string serves as the key in the sparse Q-table dictionary.

## State Space Size

**Theoretical total states:**
```
3 (pitch) × 3 (roll) × 3 (vx) × 3 (vy) × 4 (altitude) × 
6 (cx) × 6 (cy) × 2 (conf) × 3 (dx) × 3 (dy) = 209,952 states
```

**Practical implementation:**
- Uses a sparse dictionary (only stores visited state-action pairs)
- Memory efficient: only allocates Q-values for states actually encountered
- Typical training might explore 1-5% of theoretical state space

## EMA Smoothing

Detection coordinates (cx, cy) use Exponential Moving Average (EMA) smoothing to reduce jitter:
```
smooth_cx = 0.3 * new_cx + 0.7 * smooth_cx
smooth_cy = 0.3 * new_cy + 0.7 * smooth_cy
```
This helps stabilize the state representation when detections fluctuate frame-to-frame.

## Design Rationale

### Why these discretization granularities?

1. **Coarse attitude bins (3)**: Drone attitude changes slowly; fine granularity not needed
2. **Coarse velocity bins (3)**: Action space already controls velocity; fine state resolution adds complexity without benefit
3. **Medium altitude bins (4)**: Important for landing behavior at different heights
4. **Fine position bins (6)**: Critical for precise alignment with landing pad
5. **Binary confidence (2)**: Simple detected/not-detected distinction sufficient
6. **Coarse delta bins (3)**: Motion direction more important than exact speed

### Trade-offs
- **Finer bins** → Larger state space → Slower learning, more memory
- **Coarser bins** → Smaller state space → Faster learning, but less precise control
- Current configuration balances learning speed with control precision

## Example State

```python
# Raw observations
pitch = 0.1 rad
roll = -0.05 rad
vx = 0.3 m/s
vy = -0.2 m/s
altitude = 1.5 m
cx = 0.6  # Platform right of center in image
cy = 0.4  # Platform above center in image
conf = 0.8  # High confidence detection
dx = 0.1  # Moving right in view
dy = -0.05  # Moving up in view

# After normalization and discretization
state_key = "2_1_2_1_2_3_2_1_2_1"
```

## See Also

- `discretizer.py` - Implementation of normalization and discretization functions
- `env.py` - State composition from telemetry and detection
- `qlearn.py` - Sparse Q-table implementation
- `README.md` - General project overview
