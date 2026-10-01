from manim import *
from config.theme import *
from components.plots import axis

def retention_plane():
    return axis((-5.25,-2.5),4.8,1.55,'Reasoning accuracy  →','Retention quality  ↑')

def point(reasoning,loss):
    # Monotonic mapping: lower DATA-D loss maps upward.
    return (-5.25+10.05*reasoning/100,-2.5+4.05*(1-min(max(loss-2.0,0),14)/14))
