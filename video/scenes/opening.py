from manim import *
from config.theme import *
from components.layout import FilmScene
from components.architecture import model_stack
from components.typography import text
from components.data import read
A=read('data/architecture.json')

class Opening(FilmScene):
    def construct(self):
        # Sparse, fixed latent lattice; deterministic drift rather than particles.
        nodes=VGroup()
        for i in range(7):
            for j in range(4):
                p=(-5.8+i*1.84,-2.3+j*1.55,0)
                nodes.add(Dot(p,radius=.018,color=DIM,fill_opacity=.55))
                if i>0: nodes.add(Line((p[0]-1.84,p[1],0),p,stroke_width=.6,color=DIM,stroke_opacity=.28))
        self.add(nodes)
        title=text('LatticeLM',76,WHITE,BOLD).move_to((-4.4,.38,0),aligned_edge=LEFT)
        sub=text('Architecture  ·  Training  ·  Retention',25,GRAY).next_to(title,DOWN,buff=.22,aligned_edge=LEFT)
        stack=model_stack(4.4,0,2.5,4.6)
        count=text(f"{A['parameters']:,} parameters",23,MINT).move_to((4.4,-2.72,0))
        self.play(nodes.animate.shift(RIGHT*.16),Write(title),run_time=1.8)
        self.play(Create(stack),FadeIn(sub),run_time=2.8)
        self.play(FadeIn(count),run_time=.8)
        self.wait(2.5)
        self.finish()
