#!/usr/bin/env python3
# Test script to run the trained Q-learning policy in the drone tracking environment.
"""
Test script to run the trained Q-learning policy in the drone tracking environment.
Loads the final Q-table and runs greedy policy (no exploration).
"""
import asyncio
import time
import numpy as np
from env import TrackingEnvironment, EpisodeConfig
from qlearn import QLearningTable

async def run_greedy_episode(env, q_table, max_steps=1000):
    # Start from current observed state (do not reset the drone)
    # Initialize episode timing/counters but do not change drone state
    env.episode_start_time = time.time()
    env.step_count = 0
    state = await env._get_current_state()
    total_reward = 0.0
    steps = 0
    print("\n▶️ Running greedy policy episode...")
    env._descending = True
    while True:
        action = q_table.argmax(state)
        next_state, reward, done, info = await env.step(action)
        total_reward += reward
        steps += 1
        state = next_state
        if done or steps >= max_steps:
            break
    print(f"Episode finished: steps={steps}, total_reward={total_reward:.2f}, terminal={info.get('terminal_reason')}")
    return total_reward, steps, info

async def main():
    env = TrackingEnvironment(EpisodeConfig(max_time_s=60.0))
    q_table = QLearningTable(n_actions=env.n_actions)
    loaded = q_table.load("checkpoints/qtable_episode_725.pkl")
    if not loaded:
        print("⚠️ Q-table not found or failed to load. Running with empty Q-table.")

    # Ensure MAVSDK is connected before sending offboard velocity commands
    try:
        await env.mavsdk.connect()
        print("✅ MAVSDK connect successful")
    except Exception as e:
        print(f"⚠️ MAVSDK connect failed: {e} - offboard commands may not work")

    # Run a few greedy episodes
    for i in range(3):
        await run_greedy_episode(env, q_table)

if __name__ == "__main__":
    asyncio.run(main())
