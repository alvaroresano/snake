import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
from collections import deque

# Goal length that normalizes the length feature
SNAKE_LEN_GOAL = 30
# History of actions in the observation
STACKED_ACTIONS = 5


class SnakeEnv(gym.Env):
    """Custom Snake environment with lightweight tabular observations."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        *,
        grid_size: int = 200,
        cell_size: int = 10,
        render_mode: str | None = None,
        render_fps: int = 10,
        action_size: int = 1,
        max_steps: int = 1000,
        max_steps_without_food: int | None = None,
        reward_config: dict | None = None,
    ):
        super().__init__()

        if grid_size % cell_size != 0:
            raise ValueError("grid_size must be divisible by cell_size.")

        self.grid_size = grid_size
        self.cell_size = cell_size
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.action_size = action_size
        self.max_steps = max_steps
        self.max_steps_without_food = max_steps_without_food or max_steps // 2
        self.reward_config = reward_config or {
            "apple": 20.0,
            "death": -24.0,
            "distance": 0.3,
            "step": -0.5,
        }

        self.action_space = spaces.Discrete(4)

        obs_low = np.array(
            [0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            + [-1.0] * STACKED_ACTIONS,
            dtype=np.float32,
        )
        obs_high = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.sqrt(2), np.sqrt(2)]
            + [3.0] * STACKED_ACTIONS,
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        self.img = np.zeros((self.grid_size, self.grid_size, 3), dtype="uint8")
        self.score = 0
        self.max_score = 0
        self.prev_actions: deque[int] = deque(maxlen=STACKED_ACTIONS)
        self.steps_since_reset = 0
        self.steps_since_last_apple = 0

    # Step function that updates the environment after an action is taken
    def step(self, action):
        # Store previous actions
        self.prev_actions.append(action)
        self.steps_since_reset += 1
        self.steps_since_last_apple += 1

        prev_distance = self._distance_to_apple()
        self._take_action(action)

        reward = self.reward_config["step"]
        ate_apple = self.snake_head == self.apple_position

        if ate_apple:
            self.apple_position = self._spawn_apple()
            self.snake_position.insert(0, list(self.snake_head))
            reward += self.reward_config["apple"]
            self.score += 1
            self.steps_since_last_apple = 0
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        terminated = self._collision_with_boundaries() or self._collision_with_self()
        if terminated:
            reward += self.reward_config["death"]
        else:
            current_distance = self._distance_to_apple()
            if current_distance < prev_distance:
                reward += self.reward_config["distance"]
            else:
                reward -= self.reward_config["distance"]

        truncated = (
            self.steps_since_reset >= self.max_steps
            or self.steps_since_last_apple >= self.max_steps_without_food
        )

        info = {}

        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

    # Reset the environment to the initial state
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.img = np.zeros((self.grid_size, self.grid_size, 3), dtype="uint8")

        half_table = self.grid_size // 2
        self.snake_position = [
            [half_table, half_table],
            [half_table - self.cell_size, half_table],
            [half_table - 2 * self.cell_size, half_table],
        ]

        self.apple_position = self._spawn_apple()

        if self.score > self.max_score:
            self.max_score = self.score
            print(f"New maximum score registered: {self.max_score}")

        self.score = 0
        self.snake_head = [half_table, half_table]
        self.direction = 1
        self.steps_since_reset = 0
        self.steps_since_last_apple = 0

        self.prev_actions = deque(maxlen=STACKED_ACTIONS)
        for _ in range(STACKED_ACTIONS):
            self.prev_actions.append(-1)

        observation = self._get_observation()
        return observation, {}

    # Render the game visually using OpenCV
    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode

        self._update_ui() # Update the game UI

        if mode == 'human':
            cv2.imshow('Snake Game', self.img)
            cv2.waitKey(int(1000/self.render_fps)) # in milliseconds
            
            #time.sleep(0.1)  # Add delay between frames to slow down execution
        elif mode == 'rgb_array':
            return self.img.copy()

    # Close the OpenCV windows
    def close(self):
        cv2.destroyAllWindows()

    # Update the UI with the current positions of the snake and the apple
    def _update_ui(self):
        self.img = np.zeros((self.grid_size, self.grid_size, 3), dtype="uint8")

        cv2.rectangle(
            self.img,
            (self.apple_position[0], self.apple_position[1]),
            (
                self.apple_position[0] + self.cell_size,
                self.apple_position[1] + self.cell_size,
            ),
            (0, 0, 255),
            -1
        )

        for _i, position in enumerate(self.snake_position):

            cv2.rectangle(
                self.img,
                (position[0], position[1]),
                (position[0] + self.cell_size, position[1] + self.cell_size),
                (0, 255, 255) if _i == 0 else (0, 255, 0),
                -1
            )

    # Handle actions and update the snake's position
    def _take_action(self, action):
        # Avoid direct opposite movements
        if action == 0 and self.direction == 1:
            action = 1
        elif action == 1 and self.direction == 0:
            action = 0
        elif action == 2 and self.direction == 3:
            action = 3
        elif action == 3 and self.direction == 2:
            action = 2

        # Update direction based on the action
        self.direction = action

        delta = self.cell_size * self.action_size
        if action == 0: # Left
            self.snake_head[0] -= delta
        elif action == 1: # Right
            self.snake_head[0] += delta
        elif action == 2: # Down
            self.snake_head[1] += delta
        elif action == 3: # Up
            self.snake_head[1] -= delta

    def _spawn_apple(self):
        max_cells = self.grid_size // self.cell_size
        while True:
            apple = [
                random.randrange(0, max_cells) * self.cell_size,
                random.randrange(0, max_cells) * self.cell_size,
            ]
            if apple not in self.snake_position:
                return apple

    def _collision_with_boundaries(self):
        return (
            self.snake_head[0] >= self.grid_size
            or self.snake_head[0] < 0
            or self.snake_head[1] >= self.grid_size
            or self.snake_head[1] < 0
        )

    def _collision_with_self(self):
        return self.snake_head in self.snake_position[1:]

    def _distance_to_apple(self):
        return np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

    def _get_observation(self):
        head_x = self.snake_head[0] / self.grid_size
        head_y = self.snake_head[1] / self.grid_size
        apple_delta_x = (self.apple_position[0] - self.snake_head[0]) / self.grid_size
        apple_delta_y = (self.apple_position[1] - self.snake_head[1]) / self.grid_size
        snake_length = min(len(self.snake_position) / SNAKE_LEN_GOAL, 1.0)

        body_in_direction, apple_in_direction = self._directional_sensors()
        wall_distance = self._distance_to_wall_in_direction()
        apple_distance = self._distance_to_apple() / self.grid_size
        body_distance = self._distance_to_body() / self.grid_size

        observation = np.array(
            [
                head_x,
                head_y,
                apple_delta_x,
                apple_delta_y,
                snake_length,
                body_in_direction,
                apple_in_direction,
                wall_distance,
                apple_distance,
                body_distance,
            ]
            + list(self.prev_actions),
            dtype=np.float32,
        )
        return observation

    def _direction_vector(self):
        if self.direction == 0:
            return (-1, 0)
        if self.direction == 1:
            return (1, 0)
        if self.direction == 2:
            return (0, 1)
        return (0, -1)

    def _directional_sensors(self):
        dx, dy = self._direction_vector()
        cursor = list(self.snake_head)
        apple_seen = 0.0
        body_seen = 0.0
        while True:
            cursor[0] += dx * self.cell_size
            cursor[1] += dy * self.cell_size
            if (
                cursor[0] < 0
                or cursor[0] >= self.grid_size
                or cursor[1] < 0
                or cursor[1] >= self.grid_size
            ):
                break
            if cursor == self.apple_position and not apple_seen:
                apple_seen = 1.0
            if cursor in self.snake_position[1:]:
                body_seen = 1.0
                break
        return body_seen, apple_seen

    def _distance_to_wall_in_direction(self):
        dx, dy = self._direction_vector()
        if dx == -1:
            dist = self.snake_head[0]
        elif dx == 1:
            dist = self.grid_size - (self.snake_head[0] + self.cell_size)
        elif dy == -1:
            dist = self.snake_head[1]
        else:
            dist = self.grid_size - (self.snake_head[1] + self.cell_size)
        return dist / self.grid_size

    def _distance_to_body(self):
        if len(self.snake_position) <= 1:
            return self.grid_size
        head = np.array(self.snake_head)
        body = np.array(self.snake_position[1:])
        dists = np.linalg.norm(body - head, axis=1)
        return np.min(dists)
