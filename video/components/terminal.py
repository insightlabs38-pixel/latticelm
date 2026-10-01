from manim import *
from config.theme import *

def terminal_frame():
    r=RoundedRectangle(width=4.25,height=2.85,corner_radius=.12,stroke_color=DIM,fill_color='#101923',fill_opacity=1,stroke_width=2)
    bar=Line(r.get_corner(UL)+RIGHT*.2,r.get_corner(UR)+LEFT*.2,color=DIM)
    return VGroup(r,bar)
