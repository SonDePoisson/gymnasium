import gymnasium as gym
from stable_baselines3 import PPO
import os

model_name = "Reacher"
models_dir = "models"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("Reacher-v5", max_episode_steps=500)
env.reset()

model = PPO("MlpPolicy", env, verbose=False, tensorboard_log=logdir)

TIMESTEPS = 100000
ITERS = 10
iters = 0
for i in range(ITERS):
    print(f"{i + 1}/{ITERS}")
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=f"{model_name}_PPO",
        progress_bar=True,
    )
    model.save(f"{models_dir}/{model_name}")
    print(f"Model saved in : {models_dir}/{model_name}")
