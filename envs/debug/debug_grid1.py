import gym
import random
from gym import spaces
import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

MAPS = {
    "3x3": [
        [{"r":-5, "f":True}, {"r":0, "f":False}, {"r":5, "f":True}],
        [{"r":0, "f":False}, {"r":0, "f":False}, {"r":0, "f":False}],
        [{"r":0, "f":False}, {"r":0, "f":False}, {"r":-5, "f":True}]
    ]
}

class DebugGrid1Env(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(2)
        self.world = MAPS["3x3"]
        self._reset()

    def getIndex(self, x, y):
        return self.world[3-y][x-1]

    def getReward(self, x, y):
        return self.getIndex(x,y)["r"]

    def isFinal(self, x, y):
        return self.getIndex(x,y)["f"]

    def performAction(self, action):
        if action == UP:
            self.currentY = min(self.currentY + 1, 3)
        elif action == DOWN:
            self.currentY = max(self.currentY - 1, 1)
        elif action == LEFT:
            self.currentX = max(self.currentX - 1, 1)
        elif action == RIGHT:
            self.currentX = min(self.currentX + 1, 3)
        return [self.currentX, self.currentY]

    def _step(self, action):
        assert self.action_space.contains(action)
        self.performAction(action)
        return self._get_obs(), self.getReward(self.currentX, self.currentY), self.isFinal(self.currentX, self.currentY), {}

    def _get_obs(self):
        return np.array([self.currentX, self.currentY])

    def _reset(self):
        self.currentX = 1
        self.currentY = 1
        return self._get_obs()