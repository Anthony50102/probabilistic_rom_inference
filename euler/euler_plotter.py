import sys
sys.path.append("../")
from plotter import Plotter
import matplotlib.pyplot as plot

class EulerPlotter(Plotter):
    def __init__(self) -> None:
        super().__init__()