#!/usr/bin/env python3
# RL environment: Landing drone on a stationary platform using onboard detection (images) and MAVSDK.
"""
Environment wrapper for Q-learning drone tracking in PX4 SITL + Gazebo.
Reads telemetry via MAVSDK, detection via YOLO, composes state, executes actions.
"""
# env.py: Q-learning environment for drone platform tracking in PX4 SITL + Gazebo. Handles state composition, reward, and episode logic for RL training.
import asyncio
import time
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

from mavsdk_interface import MAVSDKInterface
from yolo_interface import YOLOInterface
from discretizer import normalize_features, discretize_features, encode_state


@dataclass
class EpisodeConfig:
    """Episode configuration parameters."""
    max_time_s: float = 90.0  # Changed to time-based episodes (1 minute)
    success_radius_m: float = 0.15
    success_duration_s: float = 10.0
    detection_loss_timeout_s: float = 10.0
    detection_conf_threshold: float = 0.4
    step_rate_hz: float = 10.0


@dataclass
class ResetConfig:
    """Reset configuration for drone starting position."""
    x_min: float = -1.81
    x_max: float = 1.3
    y_min: float = -1.2
    y_max: float = 1.13
    z_const: float = 2.2



class TrackingEnvironment:
    """Q-learning environment for drone platform tracking."""

    def __init__(self, episode_config: EpisodeConfig = None, reset_config: ResetConfig = None):
        self.episode_config = episode_config or EpisodeConfig()
        self.reset_config = reset_config or ResetConfig()

        # Initialize interfaces
        self.mavsdk = MAVSDKInterface()
        self.yolo = YOLOInterface()

        # Action space: 7x7 velocity combinations
        self.velocity_values = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
        self.n_actions = len(self.velocity_values) ** 2

        # State tracking
        self.episode_start_time = 0.0
        self.step_count = 0
        self.last_detection_time = 0.0
        self.last_cx = 0.5  # Default center if no detection
        self.last_cy = 0.5
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.last_distance = float('inf')
        self.success_start_time = None
        self.last_action_index = 0

        # EMA smoothing for cx, cy
        self.ema_alpha = 0.3
        self.smooth_cx = 0.5
        self.smooth_cy = 0.5

        print("✅ TrackingEnvironment initialized")

    async def reset(self) -> str:
        """Reset environment and return initial state."""
        print("🔄 Resetting episode...")

        # Generate random start position within bounds
        start_x = np.random.uniform(self.reset_config.x_min, self.reset_config.x_max)
        start_y = np.random.uniform(self.reset_config.y_min, self.reset_config.y_max)
        start_z = self.reset_config.z_const

        print(f"🎯 Target reset position: ({start_x:.2f}, {start_y:.2f}, {start_z:.2f})")

        # Reset drone via MAVSDK
        await self.mavsdk.reset_to_position(start_x, start_y, start_z)

        # Give more time for reset to complete and systems to stabilize
        print("⏱️ Waiting for reset to complete...")
        await asyncio.sleep(5.0)  # 5 seconds for reset stabilization

        # Reset episode state
        self.episode_start_time = time.time()
        self.step_count = 0
        self.last_detection_time = time.time()
        self.last_cx = 0.5
        self.last_cy = 0.5
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.last_distance = float('inf')
        self.success_start_time = None
        self.last_action_index = 0
        self.smooth_cx = 0.5
        self.smooth_cy = 0.5

        # Enter automatic downward descent mode after reset
        self._descending = True

        # Get initial state after reset
        await asyncio.sleep(0.5)  # Additional brief wait
        state = await self._get_current_state()
        print(f"✅ Reset complete. Initial state: {state}")
        return state

    async def step(self, action_index: int) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute action and return (next_state, reward, done, info)."""
        self.step_count += 1

        # Map action index to velocity command
        vx_idx = action_index // len(self.velocity_values)
        vy_idx = action_index % len(self.velocity_values)
        vx = self.velocity_values[vx_idx]
        vy = self.velocity_values[vy_idx]
        vz = 0.0  # Always 0 as specified

        # Apply velocity smoothing/clipping
        max_delta_v = 0.3  # Limit velocity changes per step
        if hasattr(self, '_last_vx'):
            vx = np.clip(vx, self._last_vx - max_delta_v, self._last_vx + max_delta_v)
            vy = np.clip(vy, self._last_vy - max_delta_v, self._last_vy + max_delta_v)
        self._last_vx, self._last_vy = vx, vy

        # If we are in automatic descent mode, override horizontal velocities and set constant downward vz
        if getattr(self, '_descending', False):
            # vx, vy = 0.0, 0.0
            vz = 0.1  # constant downward velocity (m/s)
            self._last_vx, self._last_vy = vx, vy

        # Send velocity command
        await self.mavsdk.set_velocity(vx, vy, vz)

        # Wait for step duration
        await asyncio.sleep(1.0 / self.episode_config.step_rate_hz)

        # Get next state
        next_state = await self._get_current_state()

        # Compute reward and check terminals
        reward, done, info = await self._compute_reward_and_terminals(action_index)

        # If we were descending and reached bottom, exit descending mode so next reset can be performed by training loop
        if getattr(self, '_descending', False) and done:
            self._descending = False

        self.last_action_index = action_index

        return next_state, reward, done, info

    async def _get_current_state(self) -> str:
        """Compose current state from telemetry and detection."""
        # Get telemetry
        telemetry = await self.mavsdk.get_telemetry()
        pitch = telemetry.get('pitch', 0.0)
        roll = telemetry.get('roll', 0.0)
        vx = telemetry.get('vx', 0.0)
        vy = telemetry.get('vy', 0.0)
        altitude = telemetry.get('altitude', 2.0)

        # Get detection
        detection = await self.yolo.get_detection()
        cx, cy, conf = detection

        # Update detection tracking
        current_time = time.time()
        if conf > self.episode_config.detection_conf_threshold:
            self.last_detection_time = current_time
            # Apply EMA smoothing
            self.smooth_cx = self.ema_alpha * cx + (1 - self.ema_alpha) * self.smooth_cx
            self.smooth_cy = self.ema_alpha * cy + (1 - self.ema_alpha) * self.smooth_cy
            # Update deltas
            self.last_dx = self.smooth_cx - self.last_cx
            self.last_dy = self.smooth_cy - self.last_cy
            self.last_cx = self.smooth_cx
            self.last_cy = self.smooth_cy
        else:
            # Use last known position if detection lost recently
            cx, cy = self.last_cx, self.last_cy
            conf = 0.0

        # Normalize and discretize features
        features = {
            'pitch': pitch,
            'roll': roll,
            'vx': vx,
            'vy': vy,
            'altitude': altitude,
            'cx': cx,
            'cy': cy,
            'conf': conf,
            'dx': self.last_dx,
            'dy': self.last_dy
        }
        # print(f"🔍 Features: {features}")
# 
        normalized = normalize_features(features)
        discretized = discretize_features(normalized)
        state_key = encode_state(discretized)

        return state_key

    async def _compute_reward_and_terminals(self, action_index: int) -> Tuple[float, bool, Dict[str, Any]]:
        """Compute reward and check terminal conditions."""
        reward = 0.0
        done = False
        info = {}

        # Get current drone position relative to platform (0,0)
        telemetry = await self.mavsdk.get_telemetry()
        drone_x = telemetry.get('x', 0.0)
        drone_y = telemetry.get('y', 0.0)
        altitude = telemetry.get('altitude', 2.0)

        # Platform is at (0, 0)
        distance = np.sqrt(drone_x**2 + drone_y**2)

        # Per-step success reward (unchanged)
        if distance <= self.episode_config.success_radius_m:
            reward += 5.0  # +5 per step inside success radius
            print("🏆 Inside success radius")
        else:
            # Moving away penalty (only if not inside success radius)
            if distance > self.last_distance:
                reward -= 1.0

        self.last_distance = distance

        # Smoothness penalty
        action_delta = abs(action_index - self.last_action_index)
        reward -= 0.01 * action_delta

        # If we've reached the lowest point (altitude <= 0.31), end episode and give bottom reward if within radius
        if altitude <= 0.31:
            done = True
            if distance <= self.episode_config.success_radius_m:
                reward += 50.0  # Big reward at bottom when within success radius
                info['terminal_reason'] = 'success'
            else:
                info['terminal_reason'] = 'reached_bottom'

        info.update({
            'step': self.step_count,
            'distance': distance,
            'reward_components': {
                'success': 5.0 if distance <= self.episode_config.success_radius_m else 0.0,
                'distance_penalty': -1.0 if distance > self.last_distance and distance > self.episode_config.success_radius_m else 0.0,
                'smoothness': -0.01 * action_delta
            }
        })

        return reward, done, info
