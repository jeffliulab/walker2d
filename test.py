import gymnasium as gym
from stable_baselines3 import PPO
import time
import numpy as np

# LOAD MODEL（zip file）
model = PPO.load("walker2d_ppo_model.zip")

# Create an env
print("create Walker2d-v5 env...")
env = gym.make("Walker2d-v5", render_mode="human")

# Use the same env to run several times
for test_run in range(20):
    print(f"\n========================")
    print(f"No. {test_run + 1} Test.")
    print(f"========================")
    
    obs, _ = env.reset()
    total_reward = 0
    start_time = time.time()
    
    step_count = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1

        if step_count % 100 == 0:
            print(f"Finish {step_count} steps，accumulated rewards：{total_reward:.2f}")
    
    # calculate runtime
    elapsed_time = time.time() - start_time
    
    print(f"\nTest {test_run + 1} completed.")
    print(f"Total steps: {step_count}")
    print(f"Total rewards: {total_reward:.2f}")
    print(f"Runtime: {elapsed_time:.2f} seconds")


env.close()
print("\nCOMPLETE.")