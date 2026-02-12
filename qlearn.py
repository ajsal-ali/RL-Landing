#!/usr/bin/env python3
# Sparse Q-table implementation for tabular Q-learning agent, with save/load and epsilon-greedy policy.
"""
Sparse Q-table implementation for tabular Q-learning agent, with save/load and epsilon-greedy policy.
"""
import pickle
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path


class QLearningTable:
    """Sparse Q-table using Python dictionary."""

    def __init__(self, n_actions: int, default_q: float = 0.0):
        self.n_actions = n_actions
        self.default_q = default_q
        self.q_table: Dict[Tuple[str, int], float] = {}
        self.state_visits: Dict[str, int] = {}

    def get_q(self, state: str, action: int) -> float:
        """Get Q-value for (state, action) pair."""
        key = (state, action)
        return self.q_table.get(key, self.default_q)

    def set_q(self, state: str, action: int, value: float) -> None:
        """Set Q-value for (state, action) pair."""
        key = (state, action)
        self.q_table[key] = value

        # Track state visits
        self.state_visits[state] = self.state_visits.get(state, 0) + 1

    def update(self, state: str, action: int, reward: float, next_state: str, 
               alpha: float, gamma: float) -> float:
        """Update Q-value using Q-learning rule."""
        current_q = self.get_q(state, action)

        # Find max Q-value for next state
        max_next_q = self.get_max_q(next_state)

        # Q-learning update
        target = reward + gamma * max_next_q
        new_q = current_q + alpha * (target - current_q)

        self.set_q(state, action, new_q)

        return new_q - current_q  # Return Q-value change for logging

    def get_max_q(self, state: str) -> float:
        """Get maximum Q-value for given state."""
        max_q = self.default_q
        for action in range(self.n_actions):
            q_val = self.get_q(state, action)
            max_q = max(max_q, q_val)
        return max_q

    def argmax(self, state: str) -> int:
        """Get action with highest Q-value for given state."""
        best_action = 0
        best_q = self.get_q(state, 0)

        for action in range(1, self.n_actions):
            q_val = self.get_q(state, action)
            if q_val > best_q:
                best_q = q_val
                best_action = action

        return best_action

    def get_action_values(self, state: str) -> np.ndarray:
        """Get all Q-values for given state as numpy array."""
        return np.array([self.get_q(state, a) for a in range(self.n_actions)])

    def epsilon_greedy(self, state: str, epsilon: float) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return self.argmax(state)

    def save(self, filepath: str) -> None:
        """Save Q-table to disk using pickle."""
        data = {
            'q_table': self.q_table,
            'state_visits': self.state_visits,
            'n_actions': self.n_actions,
            'default_q': self.default_q
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"💾 Q-table saved to {filepath} ({len(self.q_table):,} entries)")

    def load(self, filepath: str) -> bool:
        """Load Q-table from disk."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.q_table = data['q_table']
            self.state_visits = data['state_visits']
            self.n_actions = data['n_actions']
            self.default_q = data['default_q']

            print(f"📂 Q-table loaded from {filepath} ({len(self.q_table):,} entries)")
            return True
        except FileNotFoundError:
            print(f"❌ Q-table file not found: {filepath}")
            return False
        except Exception as e:
            print(f"❌ Error loading Q-table: {e}")
            return False

    def get_stats(self) -> Dict[str, int]:
        """Get Q-table statistics."""
        return {
            'total_entries': len(self.q_table),
            'unique_states': len(self.state_visits),
            'total_state_visits': sum(self.state_visits.values()),
            'avg_visits_per_state': sum(self.state_visits.values()) / max(1, len(self.state_visits))
        }

    def get_top_states(self, n: int = 10) -> list:
        """Get most visited states."""
        sorted_states = sorted(self.state_visits.items(), key=lambda x: x[1], reverse=True)
        return sorted_states[:n]


if __name__ == "__main__":
    # Test Q-table
    q_table = QLearningTable(n_actions=49)

    # Test basic operations
    state1 = "1_2_1_1_2_3_3_1_1_1"
    state2 = "1_2_1_1_2_3_3_1_1_2"

    print(f"Initial Q(s1,a0): {q_table.get_q(state1, 0)}")

    # Update Q-value
    delta = q_table.update(state1, 0, 1.0, state2, alpha=0.1, gamma=0.99)
    print(f"After update Q(s1,a0): {q_table.get_q(state1, 0)}, delta: {delta}")

    # Test epsilon-greedy
    action = q_table.epsilon_greedy(state1, epsilon=0.1)
    print(f"Epsilon-greedy action: {action}")

    # Test save/load
    q_table.save("test_qtable.pkl")
    new_table = QLearningTable(49)
    new_table.load("test_qtable.pkl")

    print("Stats:", q_table.get_stats())
