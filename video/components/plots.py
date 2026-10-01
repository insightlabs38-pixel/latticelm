from manim import *
from config.theme import *

def dot(x,y,color=MINT,r=.08): return Dot((x,y,0),radius=r,color=color)

def smooth_path(points,color=BLUE,width=4):
    return VMobject(color=color,stroke_width=width).set_points_smoothly([list(p)+[0] for p in points])

def axis(origin,xend,yend,xlabel,ylabel):
    ox,oy=origin
    g=VGroup(Arrow((ox,oy,0),(xend,oy,0),buff=0,color=GRAY,stroke_width=2),Arrow((ox,oy,0),(ox,yend,0),buff=0,color=GRAY,stroke_width=2))
    g.add(Text(xlabel,font=FONT,font_size=20,color=GRAY).move_to(((ox+xend)/2,oy-.38,0)))
    g.add(Text(ylabel.replace('↑','').strip(),font=FONT,font_size=20,color=GRAY).rotate(PI/2).move_to((ox-.47,(oy+yend)/2,0)))
    return g
