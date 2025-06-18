# import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

model_name = "ReacherParallel"
models_dir = "models"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = make_vec_env("Reacher-v5", n_envs=8, env_kwargs={"max_episode_steps": 500})

model = PPO("MlpPolicy", env, verbose=False, tensorboard_log=logdir)

TIMESTEPS = 200000
ITERS = 10

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
