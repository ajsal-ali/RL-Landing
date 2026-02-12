#!/usr/bin/env python3
# discretizer.py: Functions to normalize and discretize continuous observations into discrete bins for RL state representation.
"""
Functions to normalize and discretize continuous observations into discrete bins.
"""
import numpy as np
from typing import Dict, Tuple


# Discretization configuration
BINS_CONFIG = {
    'pitch_bins': 3,    # low/zero/high
    'roll_bins': 3,
    'vx_bins': 3,
    'vy_bins': 3,
    'altitude_bins': 4,
    'cx_bins': 6,
    'cy_bins': 6,
    'conf_bins': 2,     # detected/not
    'dx_bins': 3,
    'dy_bins': 3
}

# Normalization ranges
NORM_RANGES = {
    'pitch': (-0.5, 0.5),      # radians
    'roll': (-0.5, 0.5),       # radians
    'vx': (-0.8, 0.8),         # m/s
    'vy': (-0.8, 0.8),         # m/s
    'altitude': (0.2, 2.3),    # meters
    'cx': (0.0, 1.0),          # normalized
    'cy': (0.0, 1.0),          # normalized
    'conf': (0.0, 1.0),        # confidence
    'dx': (-0.5, 0.5),         # delta normalized
    'dy': (-0.5, 0.5)          # delta normalized
}


def normalize_features(features: Dict[str, float]) -> Dict[str, float]:
    """Normalize continuous features to [0, 1] range."""
    normalized = {}

    for key, value in features.items():
        if key in NORM_RANGES:
            min_val, max_val = NORM_RANGES[key]
            # Clip and normalize to [0, 1]
            clipped = np.clip(value, min_val, max_val)
            normalized[key] = (clipped - min_val) / (max_val - min_val)
        else:
            normalized[key] = value  # Pass through unknown features

    return normalized


def discretize_features(normalized_features: Dict[str, float]) -> Dict[str, int]:
    """Discretize normalized features into bins."""
    discretized = {}

    for key, value in normalized_features.items():
        if key == 'conf':
            # Custom bins for conf: [0, 0.3) -> 0, [0.3, 1.0] -> 1
            if value < 0.3:
                discretized[key] = 0
            else:
                discretized[key] = 1
        else:
            bin_key = f"{key}_bins"
            if bin_key in BINS_CONFIG:
                n_bins = BINS_CONFIG[bin_key]
                bin_idx = int(np.clip(value * n_bins, 0, n_bins - 1))
                discretized[key] = bin_idx
            else:
                discretized[key] = int(value)

    return discretized


def encode_state(discretized_features: Dict[str, int]) -> str:
    """Encode discretized state tuple to string key for Q-table."""
    # Define consistent ordering for state encoding
    feature_order = ['pitch', 'roll', 'vx', 'vy', 'altitude', 'cx', 'cy', 'conf', 'dx', 'dy']

    state_tuple = []
    for feature in feature_order:
        if feature in discretized_features:
            state_tuple.append(discretized_features[feature])
        else:
            state_tuple.append(0)  # Default value for missing features

    return "_".join(map(str, state_tuple))


def decode_state(state_key: str) -> Dict[str, int]:
    """Decode string key back to discretized features (for debugging)."""
    feature_order = ['pitch', 'roll', 'vx', 'vy', 'altitude', 'cx', 'cy', 'conf', 'dx', 'dy']
    values = list(map(int, state_key.split('_')))

    return {feature: value for feature, value in zip(feature_order, values)}


def get_state_space_size() -> int:
    """Calculate total theoretical state space size."""
    size = 1
    for bin_count in BINS_CONFIG.values():
        size *= bin_count
    return size


if __name__ == "__main__":
    # Test discretization
    test_features = {
        'pitch': 0.1,
        'roll': -0.05,
        'vx': 0.3,
        'vy': -0.2,
        'altitude': 2.5,
        'cx': 0.6,
        'cy': 0.4,
        'conf': 0.8,
        'dx': 0.1,
        'dy': -0.05
    }

    normalized = normalize_features(test_features)
    discretized = discretize_features(normalized)
    state_key = encode_state(discretized)

    print("Original features:", test_features)
    print("Normalized:", normalized)
    print("Discretized:", discretized)
    print("State key:", state_key)
    print("Decoded:", decode_state(state_key))
    print(f"Total state space size: {get_state_space_size():,}")
