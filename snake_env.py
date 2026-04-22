import gymnasium as gym
import numpy as np
import pygame

# Snake Directions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class SnakeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.grid_size = 20
    
        # All the possible actions the snake can take: 
        # 0 = straight, 1 = turn right, 2 = turn left

        self.action_space = gym.spaces.Discrete(3)

        # array of 11 binary sensors that the agent sees:
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
        
        super(SnakeEnv, self).reset(seed=seed)

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
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake[1:]):
            terminated = True
        
        # Check if snake ate food
        ate_food = new_head == self.food
        if ate_food and not terminated:
            self.score += 1
            self.snake.insert(0, new_head)
            self._place_food()
        else:
            self.snake.insert(0, new_head)
            if not terminated:
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
        if not hasattr(self, 'screen'):
            pygame.init()
            self.cell_size = 30
            self.screen = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            )
            pygame.display.set_caption('Snake RL')
        
        # Draw background
        self.screen.fill((0, 0, 0))

        # Draw snake
        for i, segment in enumerate(self.snake):
            color = (0, 200, 0) if i > 0 else (0, 255, 0)
            pygame.draw.rect(self.screen, color, (
                segment[0] * self.cell_size,
                segment[1] * self.cell_size,
                self.cell_size,
                self.cell_size
            ))

        # Draw food
        pygame.draw.rect(self.screen, (255, 0, 0), (
            self.food[0] * self.cell_size,
            self.food[1] * self.cell_size,
            self.cell_size,
            self.cell_size
        ))

        # Draw score
        font = pygame.font.SysFont('Arial', 24)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (5, 5))

        # Update the display
        pygame.display.flip()

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

    # Check if coords are dangerous
    def _is_dangerous(self, x, y):
        return (x < 0 or x >= self.grid_size or
                y < 0 or y >= self.grid_size or
                [x, y] in self.snake[1:])

    def _get_observation(self):
        head = self.snake[0]

        # Danger values (purely the coords around snake's head)
        ahead_coords = {
            UP:    [head[0],     head[1] - 1],
            DOWN:  [head[0],     head[1] + 1],
            RIGHT: [head[0] + 1, head[1]    ],
            LEFT:  [head[0] - 1, head[1]    ],
        }

        # Convert surrounding coords of the head to directions
        # that the snake is facing using clockwise modulo 
        straight = ahead_coords[self.direction]
        right    = ahead_coords[(self.direction + 1) % 4]
        left     = ahead_coords[(self.direction - 1) % 4]

        # Danger values of each direction
        danger_straight = 1 if self._is_dangerous(straight[0], straight[1]) else 0
        danger_right    = 1 if self._is_dangerous(right[0],    right[1])    else 0
        danger_left     = 1 if self._is_dangerous(left[0],     left[1])     else 0

        # Direction values
        moving_up    = 1 if self.direction == UP    else 0
        moving_right = 1 if self.direction == RIGHT else 0
        moving_down  = 1 if self.direction == DOWN  else 0
        moving_left  = 1 if self.direction == LEFT  else 0

        # Food values
        food_up    = 1 if self.food[1] < head[1] else 0
        food_down  = 1 if self.food[1] > head[1] else 0
        food_right = 1 if self.food[0] > head[0] else 0
        food_left  = 1 if self.food[0] < head[0] else 0

        return np.array([
            danger_straight, danger_right, danger_left,
            moving_up, moving_right, moving_down, moving_left,
            food_up, food_right, food_down, food_left
        ], dtype=np.float32)


    def close(self):
        pass
    # TODO: call pygame.quit() to clean up the pygame window

