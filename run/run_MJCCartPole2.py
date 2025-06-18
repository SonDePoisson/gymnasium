import gymnasium as gym
from stable_baselines3 import PPO

model_name = "MJCCartPole2"
models_dir = "models"

env = gym.make("InvertedDoublePendulum-v5", render_mode="human")
obs, info = env.reset()

model_path = f"{models_dir}/{model_name}"
model = PPO.load(model_path, env=env)

episodes = 1

for ep in range(episodes):
    print(f"Episode {ep + 1}")
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"{rewards=}|{terminated=}|{truncated=}")
        env.render()

env.close()
