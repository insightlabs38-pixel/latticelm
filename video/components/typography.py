from manim import *
from config.theme import *

def text(s,size=BODY_SIZE,color=WHITE,weight=NORMAL,max_width=None):
    o=Text(str(s),font=FONT,font_size=size,color=color,weight=weight)
    if max_width is not None and o.width>max_width: raise ValueError(f'Text too wide ({o.width:.2f}>{max_width}): {s}')
    return o

def number(value,label,color=WHITE,size=48):
    return VGroup(text(value,size,color,BOLD),text(label,22,GRAY)).arrange(DOWN,buff=.14,aligned_edge=LEFT)
