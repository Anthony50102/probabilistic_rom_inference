import sys
sys.path.append("../")
from plotter import plotter
import matplotlib.pyplot as plot

class euler_plotter(plotter):
    def __init__(self) -> None:
        super().__init__()