from stable_baselines3 import PPO
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from custom_env.SnekEnv import SnekEnv

model_name = "Snek"
models_dir = "models"

env = SnekEnv(render_mode="human")
obs, info = env.reset()

model_path = f"{models_dir}/{model_name}"
model = PPO.load(model_path, env=env)

episodes = 3

for ep in range(episodes):
    print(f"Episode {ep + 1}")
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print(f"Score : {len(env.snake_position) - 3}")

env.close()
