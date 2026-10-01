from manim import *
from config.theme import *


def model_stack(x=0,y=0,w=2.35,h=4.0,layers=12):
    out=VGroup()
    for i in range(layers):
        yy=y-h/2+(i+.5)*h/layers
        c=BLUE if i%2==0 else '#416DB8'
        out.add(RoundedRectangle(width=w,height=h/layers-.045,corner_radius=.045,
                                 fill_color=c,fill_opacity=.48,stroke_color=c,stroke_width=1.2).move_to((x,yy,0)))
    return out


def gqa_group(center,index,y=0,small=False):
    """Three separate queries read one physical K and V pair."""
    scale=.75 if small else 1.0
    group=VGroup()
    frame=RoundedRectangle(width=5.25*scale,height=3.25*scale,corner_radius=.18*scale,
                           stroke_color=DIM,stroke_width=1.2,fill_color=BLUE,fill_opacity=.035)
    frame.move_to((center,y,0));group.add(frame)
    q_y=y+1.0*scale;bus_y=y+.14*scale;kv_y=y-.77*scale
    qxs=[center+d*scale for d in (-1.35,0,1.35)]
    group.add(Line((qxs[0],bus_y,0),(qxs[-1],bus_y,0),color=BLUE,stroke_width=3))
    for j,x in enumerate(qxs):
        group.add(Line((x,q_y-.31*scale,0),(x,bus_y,0),color=BLUE,stroke_width=2.5))
        c=Circle(radius=.34*scale,color=BLUE,fill_color=BLUE,fill_opacity=.18,stroke_width=2).move_to((x,q_y,0))
        group.add(c,Text(f'Q{index*3+j}',font=FONT,font_size=24*scale,color=WHITE).move_to(c))
    group.add(Line((center,bus_y,0),(center,kv_y+.34*scale,0),color=MINT,stroke_width=3))
    for dx,label in ((-.63,'K'),(.63,'V')):
        x=center+dx*scale
        group.add(Line((center,kv_y+.34*scale,0),(x,kv_y+.34*scale,0),color=MINT,stroke_width=2.5))
        box=RoundedRectangle(width=.83*scale,height=.67*scale,corner_radius=.1*scale,color=MINT,
                             fill_color=MINT,fill_opacity=.15,stroke_width=2).move_to((x,kv_y,0))
        group.add(box,Text(label,font=FONT,font_size=25*scale,color=MINT).move_to(box))
    group.add(Text('3 Q  /  1 K  /  1 V',font=FONT,font_size=20*scale,color=GRAY).move_to((center,y-1.38*scale,0)))
    return group


def gqa(width=8,y=0):
    return VGroup(gqa_group(-2.85,0,y),gqa_group(2.85,1,y))
