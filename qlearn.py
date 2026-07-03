#!/usr/bin/env python3
# Sparse Double Q-table implementation for tabular Double Q-Learning agent, with save/load and epsilon-greedy policy.
"""
Sparse Double Q-table implementation for tabular Double Q-Learning agent (Hasselt, 2010).

Two independent estimators Q_A and Q_B of the SAME value function are maintained to
avoid the overestimation bias of standard Q-learning. Each timestep, a coin flip
decides which table is updated; the other table evaluates the greedily-selected
action. Action selection uses the combined estimate Q_A + Q_B.

State and action spaces are identical to the previous single-table implementation.
"""
import pickle
import numpy as np
from typing import Dict, Tuple
from pathlib import Path


class QLearningTable:
    """Sparse Double Q-table using two Python dictionaries (Q_A and Q_B)."""

    def __init__(self, n_actions: int, default_q: float = 0.0):
        self.n_actions = n_actions
        self.default_q = default_q
        # Two independent estimators of the same value function.
        self.q_a: Dict[Tuple[str, int], float] = {}
        self.q_b: Dict[Tuple[str, int], float] = {}
        self.state_visits: Dict[str, int] = {}

    @property
    def size(self) -> int:
        """Total number of stored (state, action) entries across Q_A and Q_B."""
        return len(self.q_a) + len(self.q_b)

    # --- low-level accessors -------------------------------------------------

    def _get(self, table: Dict[Tuple[str, int], float], state: str, action: int) -> float:
        """Get Q-value for (state, action) from a specific table."""
        return table.get((state, action), self.default_q)

    def get_q(self, state: str, action: int) -> float:
        """Get combined Q-value (Q_A + Q_B) for (state, action) pair."""
        key = (state, action)
        return self.q_a.get(key, self.default_q) + self.q_b.get(key, self.default_q)

    def _argmax_table(self, table: Dict[Tuple[str, int], float], state: str) -> int:
        """Action with highest value in a specific table (for selection step)."""
        best_action = 0
        best_q = self._get(table, state, 0)
        for action in range(1, self.n_actions):
            q_val = self._get(table, state, action)
            if q_val > best_q:
                best_q = q_val
                best_action = action
        return best_action

    # --- Double Q-learning update -------------------------------------------

    def update(self, state: str, action: int, reward: float, next_state: str,
               alpha: float, gamma: float) -> float:
        """Update Q-value using the Double Q-learning rule.

        With probability 0.5 update Q_A using Q_B to evaluate the action selected
        by Q_A; otherwise do the symmetric update on Q_B. Returns the change in the
        updated table's Q-value for logging.
        """
        key = (state, action)

        if np.random.random() < 0.5:
            update_table, eval_table = self.q_a, self.q_b
        else:
            update_table, eval_table = self.q_b, self.q_a

        current_q = update_table.get(key, self.default_q)

        # Select best next action with the table being updated ...
        best_next_action = self._argmax_table(update_table, next_state)
        # ... but evaluate it with the OTHER table (decorrelates selection/evaluation).
        eval_next_q = self._get(eval_table, next_state, best_next_action)

        target = reward + gamma * eval_next_q
        new_q = current_q + alpha * (target - current_q)

        update_table[key] = new_q

        # Track state visits.
        self.state_visits[state] = self.state_visits.get(state, 0) + 1

        return new_q - current_q  # Return Q-value change for logging

    def get_max_q(self, state: str) -> float:
        """Get maximum combined Q-value for given state."""
        max_q = self.default_q
        for action in range(self.n_actions):
            q_val = self.get_q(state, action)
            max_q = max(max_q, q_val)
        return max_q

    def argmax(self, state: str) -> int:
        """Get action with highest combined Q-value (Q_A + Q_B) for given state."""
        best_action = 0
        best_q = self.get_q(state, 0)

        for action in range(1, self.n_actions):
            q_val = self.get_q(state, action)
            if q_val > best_q:
                best_q = q_val
                best_action = action

        return best_action

    def get_action_values(self, state: str) -> np.ndarray:
        """Get all combined Q-values for given state as numpy array."""
        return np.array([self.get_q(state, a) for a in range(self.n_actions)])

    def epsilon_greedy(self, state: str, epsilon: float) -> int:
        """Select action using epsilon-greedy policy on the combined estimate."""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return self.argmax(state)

    def save(self, filepath: str) -> None:
        """Save both Q-tables to disk using pickle."""
        data = {
            'version': 2,
            'q_a': self.q_a,
            'q_b': self.q_b,
            'state_visits': self.state_visits,
            'n_actions': self.n_actions,
            'default_q': self.default_q
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        entries = len(self.q_a) + len(self.q_b)
        print(f"💾 Double Q-table saved to {filepath} ({entries:,} entries across Q_A + Q_B)")

    def load(self, filepath: str) -> bool:
        """Load Q-tables from disk.

        Supports both the new two-table format (version 2) and the legacy
        single-table format, which seeds both Q_A and Q_B from the old table.
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.n_actions = data['n_actions']
            self.default_q = data['default_q']
            self.state_visits = data['state_visits']

            if 'q_a' in data and 'q_b' in data:
                # New Double Q-learning format.
                self.q_a = data['q_a']
                self.q_b = data['q_b']
                entries = len(self.q_a) + len(self.q_b)
                print(f"📂 Double Q-table loaded from {filepath} ({entries:,} entries)")
            else:
                # Legacy single-table checkpoint: seed both estimators from it.
                legacy = data['q_table']
                self.q_a = dict(legacy)
                self.q_b = dict(legacy)
                print(f"📂 Legacy Q-table loaded from {filepath} "
                      f"({len(legacy):,} entries) — seeded into both Q_A and Q_B")
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
            'q_a_entries': len(self.q_a),
            'q_b_entries': len(self.q_b),
            'total_entries': len(self.q_a) + len(self.q_b),
            'unique_states': len(self.state_visits),
            'total_state_visits': sum(self.state_visits.values()),
            'avg_visits_per_state': sum(self.state_visits.values()) / max(1, len(self.state_visits))
        }

    def get_top_states(self, n: int = 10) -> list:
        """Get most visited states."""
        sorted_states = sorted(self.state_visits.items(), key=lambda x: x[1], reverse=True)
        return sorted_states[:n]


if __name__ == "__main__":
    # Test Double Q-table
    q_table = QLearningTable(n_actions=49)

    # Test basic operations
    state1 = "1_2_1_1_2_3_3_1_1_1"
    state2 = "1_2_1_1_2_3_3_1_1_2"

    print(f"Initial Q(s1,a0): {q_table.get_q(state1, 0)}")

    # Update Q-value repeatedly (coin flip decides Q_A vs Q_B each time)
    for _ in range(5):
        delta = q_table.update(state1, 0, 1.0, state2, alpha=0.1, gamma=0.99)
    print(f"After updates Q(s1,a0): {q_table.get_q(state1, 0):.4f}, last delta: {delta:.4f}")

    # Test epsilon-greedy
    action = q_table.epsilon_greedy(state1, epsilon=0.1)
    print(f"Epsilon-greedy action: {action}")

    # Test save/load
    q_table.save("test_qtable.pkl")
    new_table = QLearningTable(49)
    new_table.load("test_qtable.pkl")

    print("Stats:", q_table.get_stats())
