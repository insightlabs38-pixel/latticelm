from manim import *
from config.theme import *
from components.layout import FilmScene,top_label,status
from components.typography import text
from components.data import read
D=read('data/final_metrics.json')['recovery']
ROWS=[('BASE',D['base'],BLUE),('10M · 5% blend',D['recovery_10m_5pct'],MINT),('10M · 17% blend',D['recovery_10m_17pct'],RED),('Mixed · 15%',D['mixed_500k_15pct'],'#C8ACF0')]


def strategy(scene):
    h=top_label('04 / recovery','Can a specialist regain retention?','TWO FOLLOW-UPS')
    divider=Line((0,-2.65,0),(0,1.85,0),color=DIM,stroke_width=1.5)
    left_title=text('10M causal replay',31,WHITE,BOLD).move_to((-5.65,1.65,0),aligned_edge=LEFT)
    right_title=text('500k mixed objective',31,WHITE,BOLD).move_to((.47,1.65,0),aligned_edge=LEFT)
    left=VGroup()
    for x,label,color in [(-5.1,'JOINT',BLUE),(-3.12,'DATA-D',MINT),(-1.05,'blend',WHITE)]:
        c=Circle(radius=.36,color=color,stroke_width=2,fill_color=color,fill_opacity=.13).move_to((x,.25,0))
        left.add(c,text(label,20,color,BOLD).move_to((x,-.55,0)))
    left.add(Arrow((-4.68,.25,0),(-3.56,.25,0),buff=0,color=GRAY,stroke_width=2),Arrow((-2.7,.25,0),(-1.48,.25,0),buff=0,color=MINT,stroke_width=2))
    left.add(text('100% DATA-D',24,MINT,BOLD).move_to((-3.1,-1.45,0)),text('constant 3e-5 LR',22,GRAY).move_to((-3.1,-1.92,0)))
    right=VGroup()
    joint=Circle(radius=.36,color=BLUE,stroke_width=2,fill_color=BLUE,fill_opacity=.13).move_to((1.25,.25,0))
    combined=Circle(radius=.36,color=MINT,stroke_width=2,fill_color=MINT,fill_opacity=.13).move_to((5.1,.25,0))
    right.add(joint,combined,text('JOINT',20,BLUE,BOLD).move_to((1.25,-.55,0)),text('blend',20,MINT,BOLD).move_to((5.1,-.55,0)))
    right.add(Arrow((1.68,.25,0),(2.55,.75,0),buff=0,color=BLUE,stroke_width=2),Arrow((1.68,.25,0),(2.55,-.15,0),buff=0,color=MINT,stroke_width=2))
    right.add(text('reasoning',21,BLUE).move_to((3.35,.95,0)),text('20% DATA-D',21,MINT).move_to((3.35,-1.03,0)))
    right.add(Arrow((4.16,.75,0),(4.7,.3,0),buff=0,color=BLUE,stroke_width=2),Arrow((4.16,-.15,0),(4.7,.2,0),buff=0,color=MINT,stroke_width=2))
    right.add(text('500k continued tokens',22,GRAY).move_to((3.28,-1.92,0)))
    takeaway=text('Two paths back toward retention  ·  both require full-suite comparison',22,WHITE).move_to((0,-2.82,0))
    scene.add(h,divider,left_title,right_title,left,right,takeaway,status())
    return VGroup(h,divider,left_title,right_title,left,right,takeaway)


def comparison(scene):
    h=top_label('04 / recovery','Proxy gain did not predict broad gain','FULL GIBC CHECK')
    xs=[-2.35,.15,2.58,5.55]
    headers=['DATA-D','WIKI BPB','PROXY','GIBC']
    colors=[GRAY,GRAY,BLUE,MINT]
    heads=VGroup(*[text(label,19,c,BOLD).move_to((x,1.65,0),aligned_edge=RIGHT) for label,x,c in zip(headers,xs,colors)])
    line=Line((-5.95,1.25,0),(5.95,1.25,0),color=DIM,stroke_width=1.5)
    rows=VGroup();rules=VGroup()
    for i,(name,d,color) in enumerate(ROWS):
        y=.72-i*.73
        rows.add(text(name,21,color,BOLD if i==0 else NORMAL).move_to((-5.9,y,0),aligned_edge=LEFT))
        vals=[f"{d['data_d_ratio']:.3f}×",f"{d['wikitext_bpb_ratio']:.3f}×",f"{d['reasoning_proxy']:.4f}",f"{d['gibc_mean']:.4f}"]
        for j,(x,value) in enumerate(zip(xs,vals)):
            value_color=MINT if (i==0 and j==3) else color if j==2 else WHITE
            rows.add(text(value,24,value_color,BOLD if (i==0 and j==3) else NORMAL).move_to((x,y,0),aligned_edge=RIGHT))
        rules.add(Line((-5.95,y-.34,0),(5.95,y-.34,0),color=DIM,stroke_width=1))
    ratios=text('Retention ratios relative to BASE',19,GRAY).move_to((-5.9,-2.12,0),aligned_edge=LEFT)
    divergence=VGroup(text('17% blend',22,RED,BOLD),text('proxy 0.3672 ↑',22,BLUE),text('GIBC 0.4151 ↓',22,WHITE)).arrange(RIGHT,buff=.45).move_to((-2.45,-2.52,0))
    finding=text('Specialized behavior was real; broad improvement was not demonstrated.',23,MINT,BOLD).move_to((0,-3.06,0))
    scene.add(h,heads,line,rows,rules,ratios,divergence,finding,status())
    return VGroup(h,heads,line,rows,rules,ratios,divergence,finding)

class Recovery(FilmScene):
    def construct(self):
        first=strategy(self)
        self.remove(first[0]);self.add(*first[0])
        self.wait(13.9)
        c=comparison(self);self.remove(*c)
        self.play(Succession(FadeOut(VGroup(*first[1:])),FadeIn(VGroup(*c[3:]))),FadeIn(VGroup(c[1],c[2])),Succession(FadeOut(first[0][1]),FadeIn(c[0][1])),Succession(FadeOut(first[0][2]),FadeIn(c[0][2])),run_time=1.1)
        self.wait(45-self.time)
        self.finish()
class RecoveryStill(FilmScene):
    def construct(self):comparison(self);self.finish()
