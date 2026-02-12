# Reinforcement Learning-based Autonomous Multi-Rotor Landing on Stationary Platforms

A minimal Python project that trains a tabular Q-learning agent to land a drone on a stationary platform using onboard sensing (images + telemetry).

## Overview

This repository implements a tabular Q-learning agent that controls a drone in PX4 SITL + Gazebo. The agent uses discretized onboard telemetry and YOLO visual detections to navigate, approach, and land on a stationary platform.

Key goals:
- Sparse Q-table to keep memory low
- Use MAVSDK for flight control
- YOLO for onboard visual detections
- Altitude-based episodes that end when the drone descends to just above the landing pad

---

## Demo

🎥 [Watch demo (MP4)](./demo/demo.mp4)

---

## Features (high level)

- Tabular Q-learning with a sparse dict Q-table (memory only used for visited state-action pairs)
- YOLO detection interface that subscribes to Gazebo camera topic and returns normalized center/confidence
- MAVSDK offboard velocity control with position fallback and safety clamping
- Altitude-based episodes that automatically terminate when the drone reaches a defined minimum altitude (just above the landing pad)
- EMA smoothing for detection coordinates to reduce jitter
- Checkpointing and CSV logging (`training_metrics.csv`)

---

## State / Observation

The observation is a fixed set of normalized features which are discretized into bins for the tabular agent. See `discretizer.py` for exact ranges and bins.

Important features (normalized / discretized):
- pitch, roll (rad)
- body-frame vx, vy (m/s)
- altitude (m)
- detection center cx, cy (normalized image coordinates)
- detection confidence (binary bins)
- detection delta dx, dy (center change)

Discretization (configurable in `discretizer.py`):
- pitch, roll: 3 bins
- vx, vy: 3 bins each
- altitude: 4 bins
- cx, cy: 6 bins each
- confidence: 2 bins
- dx, dy: 3 bins each

This leads to a large theoretical state space; the implementation stores Q-values sparsely.

---

## Action space

- 7 × 7 grid of horizontal velocity commands mapping to body-frame (vx, vy)
- Velocity values used: `[-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]` m/s
- Vertical velocity for training episodes is `vz = +0.1` (downwards in body frame) until a target low altitude is reached.
- A smoothing limit prevents large changes in commanded velocities between steps.

---

## Reward function and terminals

- +5 per step when the drone is within success radius (default 0.15 m)
- -1 penalty when the drone moves away from the platform (distance increases)
- -0.01 × |Δaction_index| smoothness penalty
- When altitude reaches the defined landing altitude threshold (default 0.31 m), the episode ends:
  - If the drone is within success radius at that bottom point → **+50 reward** and terminal reason `success`
  - Otherwise → terminal reason `reached_bottom`

---

## Q-learning configuration

Default hyperparameters (see `trainer.py` `TrainingConfig`):
- alpha (learning rate): 0.1
- gamma (discount): 0.99
- epsilon start/min: 1.0 → 0.05 (linear decay over configured episodes)
- Episodes continue until the drone reaches the predefined minimum altitude threshold.

---

## Installation

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate

2) Install Python dependencies:

```bash
pip install -r requirements.txt
```

3) Ensure PX4 SITL + Gazebo environment is running and camera topic is available.

---

## Quick start (training)

```bash
chmod +x run.sh
./run.sh
# or
python3 trainer.py
```

Training notes:
- Episode length: 60 seconds each
- Reset stabilization: 5 seconds after reset
- Control loop: 10 Hz

Monitor training:

```bash
tail -f training_metrics.csv
ls -la checkpoints/
```

---

## Testing the trained policy

Run the provided test script to execute a greedy policy using a saved Q-table:

```bash
python3 test.py
```

By default the test script loads `checkpoints/qtable_final.pkl` (or a named checkpoint) and runs several episodes without exploration. The test script is designed to plan from the current drone state (no reset) so the vehicle will continue from its present pose in simulation.

---

## Project structure

```
├── env.py               # Environment wrapper
├── discretizer.py       # Feature normalization and discretization
├── qlearn.py            # Sparse Q-table implementation
├── trainer.py           # Training loop and scheduling
├── test.py              # Run greedy policy using saved Q-table
├── yolo_interface.py    # YOLO detection (Gazebo camera subscriber)
├── mavsdk_interface.py  # MAVSDK control and telemetry
├── requirements.txt
├── run.sh
├── README.md
├── demo/                # optional demo video(s)
│   └── demo.mp4
├── checkpoints/         # Q-table checkpoints
└── training_metrics.csv
```

---

## Troubleshooting & tips

- Camera topic missing: verify Gazebo publishes `rgbd_image/image` or update `yolo_interface.py` topic name.
- MAVSDK not connecting / offboard errors: ensure PX4 SITL is running and the correct connection string is used (default udp://:14540). Call `env.mavsdk.connect()` and allow a short delay for plugins to initialize.
- If offboard commands fail with "Offboard plugin has not been initialized", verify System.connect() completed and that `drone.offboard` is available.
- YOLO weights: update the path in `yolo_interface.py` to point to your trained model file.

