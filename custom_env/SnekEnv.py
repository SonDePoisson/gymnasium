# Code imported from PythonProgramming BS3 Tuto (cf README.md)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
from collections import deque
import time

SNAKE_LEN_GOAL = 30


class SnekEnv(gym.Env):
    def __init__(self, render_mode=None, debug=False):
        super(SnekEnv, self).__init__()
        self.render_mode = render_mode
        self.debug = debug
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(
            low=-500, high=500, shape=(6 + SNAKE_LEN_GOAL,), dtype=np.float32
        )
        self.img = np.zeros((500, 500, 3), dtype="uint8")

    def step(self, action):
        self.terminated = False
        self.truncated = False
        self.is_apple_reward = False
        self.is_boundaries_reward = False
        self.is_self_collision_reward = False
        self.is_backward_reward = False

        # if self.debug:
        #     action_int = int(action)
        #     action_str = {0: "LEFT", 1: "RIGHT", 2: "DOWN", 3: "UP"}.get(
        #         action_int, "UNKNOWN"
        #     )
        #     # print(f"[DEBUG] Action : {action_str}")

        if len(self.prev_actions) > 0:
            prev_action = int(self.prev_actions[-1])
            curr_action = int(action)
            # 0: gauche, 1: droite, 2: bas, 3: haut
            if (
                (prev_action == 0 and curr_action == 1)
                or (prev_action == 1 and curr_action == 0)
                or (prev_action == 2 and curr_action == 3)
                or (prev_action == 3 and curr_action == 2)
            ):
                self.is_backward_reward = True
                if self.debug:
                    print("[WARNING]: Backward action !")

        self.prev_actions.append(action)

        self.button_direction = action
        # Change the head position based on the button direction
        self.__do_action()

        # Increase Snake length on eating apple
        self.__is_apple_collision()

        # On collision kill the snake and print the score
        self.__is_kill_collision()

        self.__reward()

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        score = self.score

        # create observation:
        observation = [
            head_x,
            head_y,
            apple_delta_x,
            apple_delta_y,
            snake_length,
            score,
        ] + list(self.prev_actions)
        observation = np.array(observation)

        # Affichage uniquement si render_mode == "human"
        self.__game_display()
        if self.render_mode == "human":
            self.render()

        return observation, self.reward, self.terminated, self.truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.img = np.zeros((500, 500, 3), dtype="uint8")
        # Initial Snake and Apple position
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [
            random.randrange(1, 50) * 10,
            random.randrange(1, 50) * 10,
        ]
        self.score = 0
        self.button_direction = 1
        self.snake_head = [250, 250]

        self.reward = 0
        self.total_reward = 0
        self.prev_reward = 0

        self.done = False

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        score = self.score

        self.prev_actions = deque(
            maxlen=SNAKE_LEN_GOAL
        )  # however long we aspire the snake to be
        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)  # to create history

        # create observation:
        observation = [
            head_x,
            head_y,
            apple_delta_x,
            apple_delta_y,
            snake_length,
            score,
        ] + list(self.prev_actions)
        observation = np.array(observation)

        # Affichage uniquement si render_mode == "human"
        self.__game_display()
        if self.render_mode == "human":
            self.render()

        return observation, {}

    def render(self):
        if self.render_mode == "human":
            cv2.imshow("a", self.img)
            cv2.waitKey(1)
            time.sleep(0.1)

    def close(self):
        if self.render_mode == "human":
            cv2.destroyAllWindows()

    ### PRIVATE ###

    def __game_display(self):
        self.img = np.zeros((500, 500, 3), dtype="uint8")

        # Display Apple
        cv2.rectangle(
            self.img,
            (self.apple_position[0], self.apple_position[1]),
            (self.apple_position[0] + 10, self.apple_position[1] + 10),
            (0, 0, 255),
            3,
        )
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(
                self.img,
                (position[0], position[1]),
                (position[0] + 10, position[1] + 10),
                (0, 255, 0),
                3,
            )

    def __do_action(self):
        if self.button_direction == 1:
            self.snake_head[0] += 10
        elif self.button_direction == 0:
            self.snake_head[0] -= 10
        elif self.button_direction == 2:
            self.snake_head[1] += 10
        elif self.button_direction == 3:
            self.snake_head[1] -= 10
        self.snake_position.insert(0, list(self.snake_head))

    def __create_new_apple(self):
        self.apple_position = [
            random.randrange(1, 50) * 10,
            random.randrange(1, 50) * 10,
        ]
        self.score += 1

    def __is_apple_collision(self):
        if self.snake_head == self.apple_position:
            self.__create_new_apple()
            self.is_apple_reward = True

        else:
            self.snake_position.pop()

    def __collision_with_boundaries(self):
        if (
            self.snake_head[0] >= 500
            or self.snake_head[0] < 0
            or self.snake_head[1] >= 500
            or self.snake_head[1] < 0
        ):
            if self.debug:
                print("[WARNING]: Bundaries Collision !")
            self.is_boundaries_reward = True
            return 1
        else:
            return 0

    def __collision_with_self(self):
        if self.snake_head in self.snake_position[1:]:
            if self.debug:
                print("[WARNING]: Self Collision !")
            self.is_self_collision_reward = True
            return 1
        else:
            return 0

    def __is_kill_collision(self):
        if self.__collision_with_boundaries() or self.__collision_with_self():
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500, 500, 3), dtype="uint8")
            cv2.putText(
                self.img,
                "Your Score is {}".format(self.score),
                (140, 250),
                font,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            self.terminated = True

    def __reward(self):
        apple_reward = 0
        self_collision_reward = 0
        boundaries_reward = 0
        backward_reward = 0
        step_reward = -1

        if self.is_apple_reward:
            apple_reward = 1

        if self.is_self_collision_reward:
            self_collision_reward = -1

        if self.is_boundaries_reward:
            boundaries_reward = -1

        if self.is_backward_reward:
            backward_reward = -1

        max_distance = 500 + 500
        apple_delta_x = self.apple_position[0] - self.snake_head[0]
        apple_delta_y = self.apple_position[1] - self.snake_head[1]
        distance = abs(apple_delta_x) + abs(apple_delta_y)
        distance_reward = 1 - (distance / max_distance)

        w_apple = 100
        w_self_collision = 10
        w_boundaries = 10
        w_backward = 0
        w_step = 0
        w_distance = 10

        self.total_reward = (
            w_apple * apple_reward
            + w_self_collision * self_collision_reward
            + w_boundaries * boundaries_reward
            + w_backward * backward_reward
            + w_step * step_reward
            + w_distance * distance_reward
        )
        self.reward = self.total_reward - self.prev_reward
        self.prev_reward = self.total_reward
