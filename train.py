from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from snake_env import SnakeEnv

# Create the environment, wrapped in Monitor to log training data
env = Monitor(SnakeEnv(), filename="training_log")

# Create the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train for 1,000,000 timesteps
model.learn(total_timesteps=1_000_000)

# Save the trained model
model.save("snake_model")
print("Training complete! Model saved.")