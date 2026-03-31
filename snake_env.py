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

        self.action_space = gym.spaces.Discrete(3)

        # binary array of 11 sensors that the agent sees:
        #
        # [danger_straight, danger_right, danger_left,
        # moving_up, moving_right, moving_down, moving_left,
        # food_up, food_right, food_down, food_left]

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(11,), dtype=np.float32
        )

        self.snake = None
        self.direction = None
        self.food = None
        self.score = None
    
    # Start a new game, return the initial state
    def reset(self, seed=None):
        super.reset(seed=seed)

        mid = self.grid_size // 2
        self.snake = [
            [mid, mid],
            [mid - 1, mid],
            [mid - 2, mid]
        ]
        self.direction = RIGHT
        self.score = 0
        
        if len(self.snake) < self.grid_size ** 2:
            self._place_food()
        else:
            self.food = None

        observation = self._get_observation()
        info = {}
        return observation, info


    # Take one action, return what happened
    def step(self, action):

        # Update direction
        if action == 1:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            self.direction = (self.direction - 1) % 4
        
        # Get head variable
        head = self.snake[0]

        # Get distance between head and food before movement
        distance_before = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

        # Move the snake using coords
        if self.direction == UP:
            new_head = [head[0], head[1] - 1]
        elif self.direction == DOWN:
            new_head = [head[0], head[1] + 1]
        elif self.direction == RIGHT:
            new_head = [head[0] + 1, head[1]]
        elif self.direction == LEFT:
            new_head = [head[0] - 1, head[1]]

        # Get distance between head and food after movement
        distance_after = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        
        # Check if snake died
        terminated = False
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1 >= self.grid_size] or
            new_head in self.snake[1:]):
            terminated = True
        
        # Check if snake ate food
        ate_food = new_head == self.food
        if ate_food:
            self.score += 1
            self.snake.insert(0, new_head)
            self._place_food()
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()

        # Calculate reward
        if terminated:
            reward = -10
        elif ate_food:
            reward = 10
        elif distance_after < distance_before:
            reward = 1
        else:
            reward = -1

        observation = self._get_observation()
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    # Draw the game visually
    def render(self):
        pass
    # TODO: import pygame at the top of the file
    # TODO: initialize pygame if it hasn't been initialized yet
    # TODO: create a window sized grid_size * cell_size (cell_size = 30 is reasonable)
    # TODO: draw a black background
    # TODO: draw each snake segment as a green rectangle
    # TODO: draw the head as a slightly different shade of green so it's visible
    # TODO: draw the food as a red rectangle
    # TODO: draw the score as text in the corner
    # TODO: call pygame.display.flip() to update the screen
    # TODO: handle pygame.QUIT event so the window can be closed

    def _place_food(self):
        while True:
            self.food = [
                int(self.np_random.integers(0, self.grid_size)),
                int(self.np_random.integers(0, self.grid_size))
            ]
            if self.food not in self.snake:
                break
        else:
            self.food = None

    def _get_observation(self):
        pass
    # TODO: get the current head position and direction
    # TODO: calculate danger straight, danger right, danger left
    #       (is there a wall or snake body one square in that direction?)
    # TODO: calculate moving_up, moving_right, moving_down, moving_left
    #       (just which direction are we currently facing, as 4 binary values)
    # TODO: calculate food_up, food_right, food_down, food_left
    #       (is the food in that general direction relative to the head?)
    # TODO: return all 11 values as a numpy array using np.array([...], dtype=np.float32)


    def close(self):
        pass
    # TODO: call pygame.quit() to clean up the pygame window

