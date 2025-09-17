import importlib
import sys
sys.path.append("../")
import plotter
importlib.reload(plotter)
from plotter import Plotter
import matplotlib.pyplot as plot

class EulerPlotter(Plotter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)