from stable_baselines3 import PPO
from snake_env import SnakeEnv
import pygame
import time

env = SnakeEnv()
model = PPO.load("snake_model")

observation, info = env.reset()

while True:
    action, _ = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            exit()
    
    pygame.event.pump()
    time.sleep(0.1)
    
    if terminated or truncated:
        observation, info = env.reset()