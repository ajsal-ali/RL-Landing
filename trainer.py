#!/usr/bin/env python3
# Training loop for Q-learning drone tracking agent. Handles epsilon schedule, environment stepping, metrics logging, checkpointing.
"""
Training loop for Q-learning drone tracking agent.
Handles epsilon schedule, environment stepping, metrics logging, checkpointing.
"""
import asyncio
import time
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

from env import TrackingEnvironment, EpisodeConfig, ResetConfig
from qlearn import QLearningTable


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    total_episodes: int = 1000  # Changed from total_steps to total_episodes
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_episodes: int = 800  # Decay over episodes instead of steps
    alpha: float = 0.1
    gamma: float = 0.99

    # Logging and checkpointing
    log_interval_episodes: int = 10  # More frequent logging since episodes are longer
    checkpoint_interval_episodes: int = 50
    metrics_csv_path: str = "training_metrics.csv"
    qtable_checkpoint_dir: str = "checkpoints"


class TrainingMetrics:
    """Track and log training metrics."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.episode_data: List[Dict[str, Any]] = []

        # Initialize CSV file
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'episode', 'steps', 'episode_time_s', 'total_reward', 'success', 'terminal_reason',
                'detection_uptime_pct', 'avg_confidence', 'final_distance',
                'epsilon', 'q_table_size', 'avg_q_delta'
            ])
            writer.writeheader()

    def log_episode(self, episode_data: Dict[str, Any]) -> None:
        """Log episode data to CSV."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=episode_data.keys())
            writer.writerow(episode_data)
        self.episode_data.append(episode_data)

    def get_recent_success_rate(self, window: int = 50) -> float:
        """Get success rate over recent episodes."""
        if len(self.episode_data) < window:
            window = len(self.episode_data)
        if window == 0:
            return 0.0

        recent = self.episode_data[-window:]
        successes = sum(1 for ep in recent if ep.get('success', False))
        return successes / window

    def get_recent_avg_reward(self, window: int = 50) -> float:
        """Get average reward over recent episodes."""
        if len(self.episode_data) < window:
            window = len(self.episode_data)
        if window == 0:
            return 0.0

        recent = self.episode_data[-window:]
        return sum(ep.get('total_reward', 0) for ep in recent) / window


class QLearningTrainer:
    """Q-learning trainer for drone tracking task."""

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()

        # Create environment with 1-minute episodes
        episode_config = EpisodeConfig(max_time_s=60.0)  # 1 minute episodes
        self.env = TrackingEnvironment(episode_config)

        self.q_table = QLearningTable(n_actions=self.env.n_actions)
        checkpoint_path = "checkpoints/qtable_final.pkl"
        self.q_table.load(checkpoint_path)
        self.metrics = TrainingMetrics(self.config.metrics_csv_path)

        self.total_steps = 0
        self.episode_count = 0
        self.start_time = time.time()

        print("🚀 QLearningTrainer initialized")
        print(f"   Total training episodes: {self.config.total_episodes:,}")
        print(f"   Episode duration: 60 seconds")
        print(f"   Action space size: {self.env.n_actions}")
        print(f"   Epsilon decay: {self.config.epsilon_start} → {self.config.epsilon_min} over {self.config.epsilon_decay_episodes:,} episodes")

    def get_epsilon(self) -> float:
        """Get current epsilon value with linear decay."""
        if self.episode_count >= self.config.epsilon_decay_episodes:
            return self.config.epsilon_min

        progress = self.episode_count / self.config.epsilon_decay_episodes
        return self.config.epsilon_start - progress * (self.config.epsilon_start - self.config.epsilon_min)

    async def run_episode(self) -> Dict[str, Any]:
        """Run single training episode."""
        print(f"\n🎮 Starting episode {self.episode_count + 1}")

        state = await self.env.reset()
        total_reward = 0.0
        episode_steps = 0
        q_deltas = []
        detection_times = []
        confidences = []
        episode_start_time = time.time()

        while True:
            # Select action using epsilon-greedy
            epsilon = self.get_epsilon()
            action = self.q_table.epsilon_greedy(state, epsilon)

            # Take step
            next_state, reward, done, info = await self.env.step(action)

            # Update Q-table
            q_delta = self.q_table.update(
                state, action, reward, next_state, 
                self.config.alpha, self.config.gamma
            )
            q_deltas.append(abs(q_delta))

            # Track metrics
            if 'detection_age_s' in info:
                detection_times.append(info['detection_age_s'])

            total_reward += reward
            episode_steps += 1
            self.total_steps += 1

            state = next_state

            if done:
                break

        episode_duration = time.time() - episode_start_time

        # Calculate episode metrics
        detection_uptime = sum(1 for t in detection_times if t < 1.0) / max(1, len(detection_times)) * 100
        avg_confidence = np.mean(confidences) if confidences else 0.0
        success = info.get('terminal_reason') == 'success'
        final_distance = info.get('distance', float('inf'))

        episode_data = {
            'episode': self.episode_count + 1,
            'steps': episode_steps,
            'episode_time_s': episode_duration,
            'total_reward': total_reward,
            'success': success,
            'terminal_reason': info.get('terminal_reason', 'unknown'),
            'detection_uptime_pct': detection_uptime,
            'avg_confidence': avg_confidence,
            'final_distance': final_distance,
            'epsilon': epsilon,
            'q_table_size': self.q_table.size,
            'avg_q_delta': np.mean(q_deltas) if q_deltas else 0.0
        }

        print(f"📊 Episode {self.episode_count + 1} complete:")
        print(f"   Duration: {episode_duration:.1f}s, Steps: {episode_steps}")
        print(f"   Reward: {total_reward:.1f}, Terminal: {info.get('terminal_reason')}")
        print(f"   Distance: {final_distance:.2f}m, Success: {success}")

        return episode_data

    async def train(self) -> None:
        """Main training loop."""
        print(f"🎯 Starting training for {self.config.total_episodes:,} episodes...")
        print("Each episode runs for up to 60 seconds\n")

        while self.episode_count < self.config.total_episodes:
            try:
                # Run episode
                episode_data = await self.run_episode()
                self.episode_count += 1

                # Log episode
                self.metrics.log_episode(episode_data)

                # Print progress
                if self.episode_count % self.config.log_interval_episodes == 0:
                    success_rate = self.metrics.get_recent_success_rate()
                    avg_reward = self.metrics.get_recent_avg_reward()
                    elapsed = time.time() - self.start_time

                    print(f"\n📈 Progress Report - Episode {self.episode_count:,}/{self.config.total_episodes:,}")
                    print(f"   Success rate (last 50): {success_rate:.1%}")
                    print(f"   Avg reward (last 50): {avg_reward:.1f}")
                    print(f"   Total steps: {self.total_steps:,}")
                    print(f"   Epsilon: {self.get_epsilon():.3f}")
                    print(f"   Q-table size: {self.q_table.size:,}")
                    print(f"   Elapsed: {elapsed/60:.1f}min")

                # Checkpoint Q-table
                if self.episode_count % self.config.checkpoint_interval_episodes == 0:
                    checkpoint_path = f"{self.config.qtable_checkpoint_dir}/qtable_episode_{self.episode_count}.pkl"
                    self.q_table.save(checkpoint_path)

            except KeyboardInterrupt:
                print("\n⏹️  Training interrupted by user")
                break
            except Exception as e:
                print(f"❌ Episode {self.episode_count + 1} failed: {e}")
                await asyncio.sleep(2.0)  # Brief pause before retry

        # Final save
        final_path = f"{self.config.qtable_checkpoint_dir}/qtable_final.pkl"
        self.q_table.save(final_path)

        elapsed = time.time() - self.start_time
        print(f"\n✅ Training completed!")
        print(f"   Episodes: {self.episode_count:,}")
        print(f"   Total steps: {self.total_steps:,}")
        print(f"   Elapsed: {elapsed/60:.1f}min")
        print(f"   Final success rate: {self.metrics.get_recent_success_rate():.1%}")


async def main():
    """Main training entry point."""
    # Create training configuration
    config = TrainingConfig(
        total_episodes=1200,  # Fewer episodes since each is 1 minute long
        epsilon_decay_episodes=150,
        log_interval_episodes=5,
        checkpoint_interval_episodes=25
    )

    trainer = QLearningTrainer(config)
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
