from manim import *
from config.theme import *

def equation(tex,max_width=11,min_scale=.75):
    m=MathTex(tex,font_size=42,color=WHITE)
    if m.width>max_width:
        factor=max_width/m.width
        if factor<min_scale: raise ValueError(f'Equation too wide: {tex}')
        m.scale(factor)
    return m
