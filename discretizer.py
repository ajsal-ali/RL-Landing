#!/usr/bin/env python3
# discretizer.py: Functions to normalize and discretize continuous observations into discrete bins for RL state representation.
"""
Functions to normalize and discretize continuous observations into discrete bins.

This module converts continuous drone telemetry and vision data into a discrete state
representation suitable for tabular Q-learning. The process involves:
1. Normalization: Scale continuous values to [0, 1] range
2. Discretization: Map normalized values to discrete bins
3. Encoding: Convert discretized features to string keys for Q-table lookup

For detailed explanation, see STATE_SPACE.md
"""
import numpy as np
from typing import Dict, Tuple


# Discretization configuration
# Each feature is divided into bins to create a discrete state space
BINS_CONFIG = {
    'pitch_bins': 3,    # Pitch angle: low/neutral/high tilt
    'roll_bins': 3,     # Roll angle: left/neutral/right tilt
    'vx_bins': 3,       # Forward velocity: backward/stationary/forward
    'vy_bins': 3,       # Lateral velocity: left/stationary/right
    'altitude_bins': 4, # Height: very-low/low/medium/high
    'cx_bins': 6,       # Horizontal detection position (6 zones across image width)
    'cy_bins': 6,       # Vertical detection position (6 zones across image height)
    'conf_bins': 2,     # Detection confidence: low/high (threshold at 0.3)
    'dx_bins': 3,       # Horizontal motion: moving-left/stable/moving-right
    'dy_bins': 3        # Vertical motion: moving-up/stable/moving-down
}

# Normalization ranges (min, max) for each feature
# Values outside these ranges are clipped before normalization
NORM_RANGES = {
    'pitch': (-0.5, 0.5),      # radians (~±28.6 degrees)
    'roll': (-0.5, 0.5),       # radians (~±28.6 degrees)
    'vx': (-0.8, 0.8),         # m/s (body-frame forward velocity)
    'vy': (-0.8, 0.8),         # m/s (body-frame lateral velocity)
    'altitude': (0.2, 2.3),    # meters (height above ground)
    'cx': (0.0, 1.0),          # normalized image x-coordinate (0=left, 1=right)
    'cy': (0.0, 1.0),          # normalized image y-coordinate (0=top, 1=bottom)
    'conf': (0.0, 1.0),        # YOLO detection confidence score
    'dx': (-0.5, 0.5),         # change in cx (motion in image x)
    'dy': (-0.5, 0.5)          # change in cy (motion in image y)
}


def normalize_features(features: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize continuous features to [0, 1] range.
    
    Args:
        features: Dictionary of raw feature values
        
    Returns:
        Dictionary of normalized features in [0, 1] range
        
    Process:
        1. Clip raw value to defined range (prevents outliers)
        2. Apply linear normalization: (value - min) / (max - min)
    """
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
    """
    Discretize normalized features into bins.
    
    Args:
        normalized_features: Dictionary of normalized features (values in [0, 1])
        
    Returns:
        Dictionary of discretized features (integer bin indices)
        
    Process:
        - For most features: bin_index = floor(normalized_value * n_bins)
        - For confidence: threshold-based binning at 0.3
        
    Example:
        If altitude has 4 bins and normalized_altitude = 0.7:
        bin_index = floor(0.7 * 4) = floor(2.8) = 2
    """
    discretized = {}

    for key, value in normalized_features.items():
        if key == 'conf':
            # Custom threshold-based binning for confidence
            # Low confidence [0, 0.3) -> bin 0 (not detected reliably)
            # High confidence [0.3, 1.0] -> bin 1 (detected reliably)
            if value < 0.3:
                discretized[key] = 0
            else:
                discretized[key] = 1
        else:
            bin_key = f"{key}_bins"
            if bin_key in BINS_CONFIG:
                n_bins = BINS_CONFIG[bin_key]
                # Map [0, 1] to [0, n_bins-1]
                bin_idx = int(np.clip(value * n_bins, 0, n_bins - 1))
                discretized[key] = bin_idx
            else:
                discretized[key] = int(value)

    return discretized


def encode_state(discretized_features: Dict[str, int]) -> str:
    """
    Encode discretized state tuple to string key for Q-table lookup.
    
    Args:
        discretized_features: Dictionary of discretized features (bin indices)
        
    Returns:
        String key representing the state (e.g., "1_1_2_1_2_3_3_1_1_1")
        
    Note:
        Feature order is fixed to ensure consistent state encoding.
        Missing features default to bin 0.
    """
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
    """
    Decode string key back to discretized features (for debugging/analysis).
    
    Args:
        state_key: State string (e.g., "1_1_2_1_2_3_3_1_1_1")
        
    Returns:
        Dictionary mapping feature names to bin indices
    """
    feature_order = ['pitch', 'roll', 'vx', 'vy', 'altitude', 'cx', 'cy', 'conf', 'dx', 'dy']
    values = list(map(int, state_key.split('_')))

    return {feature: value for feature, value in zip(feature_order, values)}


def get_state_space_size() -> int:
    """
    Calculate total theoretical state space size.
    
    Returns:
        Number of possible states (product of all bin counts)
        
    Note:
        The actual Q-table uses sparse storage and only allocates memory
        for visited state-action pairs. In practice, only a small fraction
        of the theoretical state space is explored during training.
    """
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
