import time
from snake_env import SnakeEnv
import pygame

env = SnakeEnv()
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            exit()

    pygame.event.pump()
    time.sleep(0.3)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()