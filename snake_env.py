import gymnasium as gym
import numpy as np

# Snake Directions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class SnakeEnv(gym.Env):
    def __innit__(self):
        super().__innit__()
        self.grid_size = 20
    
        # All the possible actions the snake can take: 
        # 0 = straight, 1 = turn right, 2 = turn left

        self.action_space() = gym.spaces.Discrete(3)

        # binary array of 11 sensors that the agent sees:
        #
        # [danger_straight, danger_right, danger_left,
        # moving_up, moving_right, moving_down, moving_left,
        # food_up, food_right, food_down, food_left]

        self.observation_space() = gym.spaces.Box(
            low=0, high=1, shape=(11,), dtype=np.float32
        )

        self.snake = None
        self.direction = None
        self.food = None
        self.score = None
    
    # Start a new game, return the initial state
    def reset(self, seed=None):
        pass

    # Take one action, return what happened
    def step(self, action):
        pass

    # Draw the game visually
    def render(self):
        pass



