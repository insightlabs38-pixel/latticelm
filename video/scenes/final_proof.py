from manim import *
from config.theme import *
from components.layout import FilmScene,top_label,status
from components.typography import text
from components.data import final


def proof(scene):
    f=final();h=top_label('05 / selection','The final selector retains BASE','VERIFICATION')
    base=VGroup(text('BASE reference',24,BLUE,BOLD),text('0.4289',48,WHITE,BOLD),text('full GIBC mean',20,GRAY)).arrange(DOWN,buff=.16,aligned_edge=LEFT).move_to((-4.85,.5,0))
    retention=VGroup(text('RETENTION GATE',20,GRAY,BOLD).move_to((-.85,1.54,0)),
                     Line((-.85,1.12,0),(-.85,-1.25,0),color=MINT,stroke_width=3),
                     text('5%  passes',22,MINT).move_to((.36,.5,0)),text('17% passes',22,MINT).move_to((.36,-.45,0)))
    retention.add(Dot((-.85,.5,0),radius=.11,color=MINT),Dot((-.85,-.45,0),radius=.11,color=MINT))
    gibc=VGroup(text('GIBC IMPROVEMENT TEST',20,GRAY,BOLD).move_to((3.62,1.54,0)),
                Line((2.15,.75,0),(5.55,.75,0),color=BLUE,stroke_width=3),
                text('BASE  0.4289',20,BLUE).move_to((3.85,1.04,0)),
                Dot((3.12,.58,0),radius=.1,color=WHITE),text('5%   0.4288',21,WHITE).move_to((4.4,.42,0)),
                Dot((3.12,-.3,0),radius=.1,color=RED),text('17%  0.4151',21,WHITE).move_to((4.4,-.43,0)))
    connection=VGroup(Arrow((-3.03,.25,0),(-1.65,.25,0),buff=0,color=GRAY,stroke_width=2.5),
                      Arrow((.95,.25,0),(2.03,.25,0),buff=0,color=GRAY,stroke_width=2.5))
    conclusion=VGroup(Line((-5.72,-1.68,0),(5.72,-1.68,0),color=DIM,stroke_width=1.5),
                      text('No evaluated candidate cleared the gain criterion.',22,WHITE).move_to((-5.7,-2.05,0),aligned_edge=LEFT),
                      text('FINAL MODEL  —  BASE',32,MINT,BOLD).move_to((5.65,-2.7,0),aligned_edge=RIGHT))
    sha=text(f"1.9B-token checkpoint  ·  SHA256 {f['final_checkpoint_sha'][:16]}…",18,GRAY).move_to((-5.7,-3.13,0),aligned_edge=LEFT)
    scene.add(h,base,retention,gibc,connection,conclusion,sha,status())
    return VGroup(h,base,retention,gibc,connection,conclusion,sha)

class FinalProof(FilmScene):
    def construct(self):
        p=proof(self);self.remove(*p);self.add(p[0],p[1])
        self.play(Create(p[4]),FadeIn(p[2]),run_time=1.2)
        self.play(FadeIn(p[3]),run_time=1.2)
        self.play(FadeIn(p[5]),FadeIn(p[6]),run_time=.8)
        self.wait(12-self.time)
        self.finish()
