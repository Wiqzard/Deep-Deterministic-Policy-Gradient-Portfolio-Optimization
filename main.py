import numpy as np
from utils.constants import NUM_ASSETS
from utils.visualize import plot_ou_action_noise
from time import sleep
import ipywidgets as widgets
from ipywidgets.interact import interact

from IPython.display import display


# plot_ou_action_noise(mu=np.zeros(1), theta=0.15, sigma=0.15, dt=0.002, x0=None, steps=500)
# sleep(10)

interact(
    plot_ou_action_noise,
    mu=(0, 10, 1),
    sigma=(0.0, 1.0, 0.01),
    theta=(0.0, 1.0, 0.01),
    x0=(0.0, 1.0, 0.01),
    dt=(0.0, 1.0, 0.01),
    steps=(1, 100, 1),
)
