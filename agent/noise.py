import numpy as np


class OUActionNoise(object):
    def __init__(self, mu:float, config, x0:float=None) -> None:
        self.theta = config.theta
        self.mu = mu
        self.sigma = config.sigma
        self.dt = config.dt
        self.x0 = config.x0
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
