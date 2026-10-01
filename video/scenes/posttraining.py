from manim import *
from config.theme import *
from components.layout import FilmScene,top_label,status
from components.typography import text
from components.data import provisional
D=provisional()


def method_node(label,x,y,color=BLUE,size=27,sub=None):
    title=text(label,size,color,BOLD).move_to((x,y,0))
    if sub:return VGroup(title,text(sub,20,GRAY).move_to((x,y-.4,0)))
    return VGroup(title)


def experiment(scene):
    h=top_label('03 / post-training','Capability is a branching experiment','METHOD SCREEN')
    rail=Line((-5.45,1.03,0),(.5,1.03,0),color=BLUE,stroke_width=3)
    core=VGroup(method_node('BASE',-5.16,1.45,MINT,29),method_node('SFT',-3.08,1.45,WHITE,29),
                method_node('screens',-.42,1.45,WHITE,29),text('optimizer · LR · replay',19,GRAY).move_to((-.42,1.89,0)))
    for x in (-5.16,-3.08,-.32):core.add(Dot((x,1.03,0),radius=.075,color=BLUE))
    branch=VGroup(Line((.5,1.03,0),(1.18,1.03,0),color=BLUE,stroke_width=3),
                  Line((1.18,.22,0),(1.18,1.7,0),color=BLUE,stroke_width=3))
    for y in (1.68,.35):branch.add(Line((1.18,y,0),(2.15,y,0),color=BLUE,stroke_width=3))
    survivors=VGroup(method_node('ranking',3.28,1.68,BLUE,30,'screened'),method_node('joint',3.28,.35,MINT,30,'specialist'))
    secondary=VGroup(text('SimPO',24,GRAY).move_to((3.28,-.75,0)),Line((1.18,.22,0),(1.18,-.75,0),color=DIM,stroke_width=2),Line((1.18,-.75,0),(2.52,-.75,0),color=DIM,stroke_width=2))
    gate=VGroup(Line((1.25,-1.47,0),(4.75,-1.47,0),color=DIM,stroke_width=2),
                text('RFT',23,GRAY).move_to((1.65,-1.8,0)),text('RLVR',23,GRAY).move_to((3.25,-1.8,0)),
                text('capability gate',20,GRAY,BOLD).move_to((4.73,-2.26,0)))
    for x in (1.65,3.25):gate.add(Line((x-.45,-2.11,0),(x+.45,-1.58,0),color=GRAY,stroke_width=2))
    evidence=VGroup(text(f"{D['base']['reasoning_accuracy']:.2f}% → {D['joint']['reasoning_accuracy']:.2f}%",29,MINT,BOLD),
                    text(f"DATA-D {D['base']['data_d']:.3f} → {D['joint']['data_d']:.3f}",23,RED)).arrange(DOWN,buff=.23,aligned_edge=LEFT).move_to((-3.55,-1.8,0))
    scope=text('Earlier 256-example specialization screen',20,GRAY).move_to((-3.55,-2.85,0))
    scene.add(h,rail,core,branch,survivors,secondary,gate,evidence,scope,status())
    return VGroup(h,rail,core,branch,survivors,secondary,gate,evidence,scope)


def frontier(scene):
    h=top_label('03 / post-training','Specialization exposes a retention cliff','EARLIER 256-EXAMPLE SCREEN')
    # Large endpoint geometry keeps the tradeoff readable without studying ticks.
    base_xy=(-4.75,.45,0);joint_xy=(3.85,-1.75,0)
    axes=VGroup(Arrow((-5.5,-2.35,0),(5.65,-2.35,0),buff=0,color=GRAY,stroke_width=2.5),
                 Arrow((-5.5,-2.35,0),(-5.5,1.8,0),buff=0,color=GRAY,stroke_width=2.5))
    axes.add(text('targeted reasoning accuracy  →',22,GRAY).move_to((0,-2.82,0)))
    axes.add(text('retention quality  ↑',22,GRAY).rotate(PI/2).move_to((-6.0,-.3,0)))
    path=Arrow(base_xy,joint_xy,buff=.3,color=BLUE,stroke_width=5)
    points=VGroup(Circle(radius=.21,color=BLUE,fill_color=BLUE,fill_opacity=.8).move_to(base_xy),
                  Circle(radius=.21,color=RED,fill_color=RED,fill_opacity=.8).move_to(joint_xy))
    base=VGroup(text('BASE',26,BLUE,BOLD),text('27.34% reasoning',22,WHITE),text('DATA-D 2.397',22,GRAY)).arrange(DOWN,buff=.13,aligned_edge=LEFT).move_to((-3.55,1.38,0))
    joint=VGroup(text('JOINT',26,RED,BOLD),text('89.45% reasoning',22,WHITE),text('DATA-D 15.025',22,GRAY)).arrange(DOWN,buff=.13,aligned_edge=LEFT).move_to((3.9,.43,0))
    finding=text('reasoning rises while retention falls',27,MINT,BOLD).move_to((-2.05,-1.55,0))
    scope=text('targeted proxy only  ·  broader capability checked separately',20,GRAY).move_to((0,-3.26,0))
    scene.add(h,axes,path,points,base,joint,finding,scope,status())
    return VGroup(h,axes,path,points,base,joint,finding,scope)

class PostTraining(FilmScene):
    def construct(self):
        a=experiment(self)
        self.remove(a[0]);self.add(*a[0])
        self.wait(18.9)
        f=frontier(self);self.remove(*f)
        self.play(Succession(FadeOut(VGroup(*a[1:])),FadeIn(VGroup(*f[2:]))),FadeIn(f[1]),Succession(FadeOut(a[0][1]),FadeIn(f[0][1])),Succession(FadeOut(a[0][2]),FadeIn(f[0][2])),run_time=1.1)
        self.play(Indicate(f[3],color=BLUE),run_time=1.6)
        self.wait(45-self.time)
        self.finish()
class PostTrainingStill(FilmScene):
    def construct(self):experiment(self);self.finish()
