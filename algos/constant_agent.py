import numpy as np

class Agent:
    def __init__(self, config, env):
        self.dims_action =  env.dims_action
        pass

    def supervised(self, replay):
        pass

    def update(self, replay):
        pass

    def act_probabilistic(self, state):
        return np.zeros(len(self.dims_action))

    def act_deterministic(self, state):
        return np.zeros(len(self.dims_action))
