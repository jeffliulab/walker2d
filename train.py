import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment
env_id = "Walker2d-v5"
env = make_vec_env(env_id, n_envs=16)  # Utilizing multiple environments for faster training

# Create the model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    tensorboard_log="./walker2d_tensorboard/",
)

# Train the model
total_timesteps = 2_000_000  # Adjust based on your available time and GPU
model.learn(total_timesteps=total_timesteps, progress_bar=True)

# Save the model
model.save("walker2d_ppo_model")

# Evaluate the trained agent
eval_env = gym.make(env_id, render_mode="rgb_array")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Test and render a few episodes
test_env = gym.make(env_id, render_mode="human")
obs, _ = test_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        obs, _ = test_env.reset()