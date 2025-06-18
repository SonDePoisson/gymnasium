import gymnasium as gym
from stable_baselines3 import PPO
import os


models_dir = "models"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make(
    "LunarLander-v3",
    continuous=True,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
)
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
        tb_log_name="LunarLander_PPO",
        progress_bar=True,
    )
    model.save(f"{models_dir}/LunarLander")
    print(f"Model saved in : {models_dir}/LunarLander")
