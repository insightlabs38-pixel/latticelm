from manim import *
from config.theme import *

class QARegistry:
    def __init__(self): self.items=[]
    def add(self,ident,obj,group=None,allow_overlap=False):
        self.items.append((ident,obj,group,allow_overlap)); return obj
    def problems(self):
        issues=[]
        for ident,obj,group,allow in self.items:
            x0,x1=obj.get_left()[0],obj.get_right()[0]; y0,y1=obj.get_bottom()[1],obj.get_top()[1]
            if x0 < -SAFE_X or x1 > SAFE_X or y0 < -SAFE_Y or y1 > SAFE_Y: issues.append(f'{ident}: outside safe frame ({x0:.2f},{x1:.2f},{y0:.2f},{y1:.2f})')
        for i,(ia,a,ga,aa) in enumerate(self.items):
            if aa: continue
            for ib,b,gb,ab in self.items[i+1:]:
                if ab or (ga and ga==gb): continue
                dx=min(a.get_right()[0],b.get_right()[0])-max(a.get_left()[0],b.get_left()[0])
                dy=min(a.get_top()[1],b.get_top()[1])-max(a.get_bottom()[1],b.get_bottom()[1])
                if dx>0.14 and dy>0.12 and dx*dy>0.08: issues.append(f'{ia} overlaps {ib} ({dx:.2f}x{dy:.2f})')
        return issues
    def overlay(self):
        m=VGroup(Rectangle(width=SAFE_X*2,height=SAFE_Y*2,color=RED,stroke_width=1))
        for ident,obj,*_ in self.items:
            box=Rectangle(width=obj.width,height=obj.height,color=MINT,stroke_width=1).move_to(obj)
            tag=Text(ident,font=FONT,font_size=12,color=MINT).next_to(box,UP,buff=.01)
            m.add(box,tag)
        return m

class FilmScene(Scene):
    def setup(self):
        self.camera.background_color=BG
        self.qa=QARegistry()
    def mark(self,ident,obj,group=None,allow_overlap=False): return self.qa.add(ident,obj,group,allow_overlap)
    def finish(self):
        import os
        if os.getenv('LATTICE_QA')=='1' or os.getenv('LATTICE_STRICT')=='1':
            seen={id(item[1]) for item in self.qa.items}
            for top in self.mobjects:
                for obj in top.get_family():
                    if isinstance(obj,(Text,MathTex)) and id(obj) not in seen:
                        seen.add(id(obj));self.qa.add(f'text_{len(self.qa.items)}',obj)
        if os.getenv('LATTICE_QA')=='1': self.add(self.qa.overlay())
        if os.getenv('LATTICE_STRICT')=='1':
            problems=self.qa.problems()
            if problems: raise AssertionError('; '.join(problems))


def rule(x1,y1,x2,y2,color=DIM,width=2):
    return Line((x1,y1,0),(x2,y2,0),color=color,stroke_width=width)

def top_label(kicker,title,section):
    k=Text(kicker.upper(),font=FONT,font_size=19,color=MINT).move_to((CONTENT_LEFT,SECTION_Y,0),aligned_edge=LEFT)
    t=Text(title,font=FONT,font_size=TITLE_SIZE,weight=BOLD,color=WHITE)
    if t.width>CONTENT_RIGHT-CONTENT_LEFT:t.scale_to_fit_width(CONTENT_RIGHT-CONTENT_LEFT)
    t.move_to((CONTENT_LEFT,TITLE_Y,0),aligned_edge=LEFT)
    n=Text(section,font=FONT,font_size=19,color=GRAY).move_to((CONTENT_RIGHT,SECTION_Y,0),aligned_edge=RIGHT)
    return VGroup(k,t,n,rule(CONTENT_LEFT,RULE_Y,CONTENT_RIGHT,RULE_Y))

def retitle(scene,header,title,section=None,run_time=.6):
    """Keep the section rail fixed while its subject changes."""
    nxt=top_label('',title,section or '')
    animations=[Transform(header[1],nxt[1])]
    if section is not None:animations.append(Transform(header[2],nxt[2]))
    scene.play(*animations,run_time=run_time)

def ambient_lines(y_values=(-2.6,-1.3,.0,1.3),opacity=.08):
    return VGroup(*[Line((CONTENT_LEFT,y,0),(CONTENT_RIGHT,y,0),color=DIM,stroke_width=1,stroke_opacity=opacity) for y in y_values])

def status():
    from components.data import is_provisional
    return Text('PROVISIONAL  /  FINAL SELECTION PENDING',font=FONT,font_size=17,color=GRAY).move_to((0,-3.43,0)) if is_provisional() else VGroup()
