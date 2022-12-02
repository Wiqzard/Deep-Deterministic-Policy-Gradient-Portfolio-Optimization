import numpy as np


class OUActionNoise(object):
    def __init__(self, mu:float, args, x0:float=None) -> None:
        self.theta = args.theta
        self.mu = mu
        self.sigma = args.sigma
        self.dt = args.dt
        self.x0 = args.x0
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
