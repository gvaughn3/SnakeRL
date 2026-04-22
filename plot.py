import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the training log
df = pd.read_csv("training_log.monitor.csv", skiprows=1)
df.columns = ["reward", "length", "time"]

# Smooth the data with a rolling average
window = 50
df["reward_smooth"] = df["reward"].rolling(window).mean()
df["length_smooth"] = df["length"].rolling(window).mean()

# Plot 1: Reward over episodes
plt.figure(figsize=(10, 5))
plt.plot(df["reward"], alpha=0.3, color="blue", label="Raw")
plt.plot(df["reward_smooth"], color="blue", label="Smoothed (50 ep)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Episode During Training")
plt.legend()
plt.savefig("reward_plot.png")
plt.close()

# Plot 2: Survival time over episodes
plt.figure(figsize=(10, 5))
plt.plot(df["length"], alpha=0.3, color="green", label="Raw")
plt.plot(df["length_smooth"], color="green", label="Smoothed (50 ep)")
plt.xlabel("Episode")
plt.ylabel("Steps Survived")
plt.title("Survival Time per Episode During Training")
plt.legend()
plt.savefig("survival_plot.png")
plt.close()

print("Plots saved as reward_plot.png and survival_plot.png")