import numpy as np
from stable_baselines3 import PPO
from snake_env import SnakeEnv

env = SnakeEnv()
model = PPO.load("snake_model")

scores = []
survival_times = []

for episode in range(100):
    observation, info = env.reset()
    terminated = False
    truncated = False
    steps = 0
    
    while not terminated and not truncated:
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
    
    scores.append(env.score)
    survival_times.append(steps)
    print(f"Episode {episode + 1}: Score = {env.score}, Steps = {steps}")

print(f"\nAverage Score: {np.mean(scores):.2f}")
print(f"Average Survival Time: {np.mean(survival_times):.2f} steps")
print(f"Max Score: {max(scores)}")
print(f"Score Std Dev: {np.std(scores):.2f}")