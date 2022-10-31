import matplotlib.pyplot as plt
import numpy as np

ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
ACTIONS_NAMES = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}


class GRID(object):

    def __init__(self, grid_size=16, max_time=2000, random=False, pixel_size=10):

        self.grid_size = grid_size
        self.max_time = max_time
        self.random = random
        self.pixel_size = pixel_size

        self.action_shape = (4,)
        self.state_shape = (self.grid_size, self.grid_size, 3)

        self.wall = np.zeros((self.grid_size, self.grid_size))
        self.setup()

        self.agent_x = self.agent_y = None
        self.target_x = self.target_y = None
        self.t = None
        self.episode = []

        self.reset()

    def setup(self):
        half = self.grid_size // 2
        self.wall[0, :] = self.wall[:, -1] = 1
        self.wall[half, :] = 1
        self.wall[half, half // 2 - 1:half // 2 + 2] = 0
        self.wall[half, -(half // 2 + 2):-(half // 2 - 1)] = 0
        self.wall = np.maximum(self.wall, self.wall.T)

    def reset(self):
        """This function resets the game and returns the initial state"""
        self.episode = []
        self.t = 0
        self.add_agent()
        self.add_target()
        return self.current_state

    def step(self, action):
        """
        :param action: 0:UP, 1:RIGHT, 2: DOWN, 3:LEFT
        :return: (new state, reward, game_over)
        """

        reward = 0
        old_x, old_y = self.agent_position
        episode_ended = self.game_over or (not self.random and self.target_found)

        if not episode_ended:
            if action in ACTIONS:
                a = ACTIONS[action]
                self.agent_x = self.agent_x + a[0]
                self.agent_y = self.agent_y + a[1]
                if self.target_found:
                    reward = 1
                if self.wall[self.agent_x, self.agent_y] == 1:
                    self.agent_x, self.agent_y = old_x, old_y
                    reward = -1
            else:
                RuntimeError('Error: action {} not recognized'.format(action))

            self.t = self.t + 1
            self.episode.append(self.get_screen())
            if self.target_found:
                self.add_target()

        return self.current_state, reward, episode_ended


    def add_agent(self):
        if self.random:
            self.agent_x, self.agent_y = 0, 0
            while self.wall[self.agent_x, self.agent_y] != 0:
                self.agent_x, self.agent_y = np.random.randint(1, self.grid_size - 1, 2)
        else:
            self.agent_x = 3
            self.agent_y = 3

    def add_target(self):

        if self.random:
            self.target_x, self.target_y = self.agent_x, self.agent_y
            while self.target_found or self.wall[self.target_x, self.target_y] != 0:
                self.target_x, self.target_y = np.random.randint(1, self.grid_size - 1, 2)
        else:
            self.target_x = self.target_y = self.grid_size - 3

    @property
    def current_state(self):
        if self.random:
            return self.get_screen(size=1)/255.0
        else:
            return self.agent_position

    @property
    def game_over(self):
        return self.t > self.max_time

    @property
    def target_found(self):
        return (self.target_x, self.target_y) == (self.agent_x, self.agent_y)

    @property
    def target_position(self):
        return (self.target_x, self.target_y)

    @property
    def agent_position(self):
        return (self.agent_x, self.agent_y)

    def get_screen(self, size=None):
        step = self.pixel_size
        if size is not None:
            step = size
        screen = np.zeros((self.state_shape[0] * step, self.state_shape[1] * step, 3), dtype="int32")

        wall_x, wall_y = np.where(self.wall == 1)
        # print(wall_x, wall_y)
        for x, y in zip(wall_x, wall_y):
            screen[x * step:(x + 1) * step][:, y * step:(y + 1) * step, 2] = 255

        screen[self.target_x * step: (self.target_x + 1) * step][:,
        self.target_y * step: (self.target_y + 1) * step] = [255, 0, 0]
        screen[self.agent_x * step: (self.agent_x + 1) * step][:,
        self.agent_y * step: (self.agent_y + 1) * step] = [255, 255, 255]
        return screen

    def render(self):
        plt.imshow(self.get_screen(size=1))
