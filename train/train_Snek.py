import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from custom_env.SnekEnv import SnekEnv

model_name = "Snek"
models_dir = "models"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = make_vec_env(SnekEnv, n_envs=8)
env.reset()

model = PPO("MlpPolicy", env, verbose=False, tensorboard_log=logdir)

TIMESTEPS = 100000
ITERS = 100

for iter in range(ITERS):
    print(f"{iter + 1}/{ITERS}")
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=f"{model_name}_PPO",
        progress_bar=True,
    )
    model.save(f"{models_dir}/{model_name}")
    print(f"Model saved in : {models_dir}/{model_name}")
