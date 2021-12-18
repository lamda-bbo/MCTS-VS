import numpy as np
import gym
from .filter import get_filter


class LinearPolicy:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.