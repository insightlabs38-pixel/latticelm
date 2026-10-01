from manim import *
from config.theme import *
from components.layout import FilmScene,status
from components.typography import text
from components.data import final


def metric(value,label,x,y,color=WHITE,size=48):
    return VGroup(text(value,size,color,BOLD).move_to((x,y,0)),text(label,21,GRAY).move_to((x,y-.49,0)))


def recap(scene):
    f=final()
    top=VGroup(text('LatticeLM',61,WHITE,BOLD).move_to((CONTENT_LEFT,3.04,0),aligned_edge=LEFT),
               text('FINAL TECHNICAL RECAP',19,MINT).move_to((CONTENT_RIGHT,3.04,0),aligned_edge=RIGHT),
               Line((CONTENT_LEFT,2.57,0),(CONTENT_RIGHT,2.57,0),color=DIM,stroke_width=1.6))
    primary=VGroup(metric('48.6M','parameters',-4.32,1.76,MINT,57),
                   metric('1.9B','pretraining tokens',0,1.76,WHITE,57),
                   metric('12 × 552','Co4 stack',4.32,1.76,WHITE,57))
    separators=VGroup(Line((-2.18,.98,0),(-2.18,2.05,0),color=DIM,stroke_width=1),
                      Line((2.18,.98,0),(2.18,2.05,0),color=DIM,stroke_width=1))
    technical=VGroup(text('6Q / 2KV real GQA    ·    QK-Norm',23,WHITE).move_to((-3.0,.52,0)),
                     text('Muon + AdamW    ·    WSD 2 / 83 / 15',23,BLUE).move_to((3.05,.52,0)))
    line1=Line((CONTENT_LEFT,.12,0),(CONTENT_RIGHT,.12,0),color=DIM,stroke_width=1)
    results=VGroup(metric(f"{f['data_d']:.4f}",'validation loss',-4.2,-.54,MINT,36),
                   metric(f"{f['wikitext_bpb']:.4f}",'WikiText BPB',0,-.54,WHITE,36),
                   metric(f"{f['wikitext_ppl']:.2f}",'WikiText PPL',4.2,-.54,WHITE,36))
    benchmarks=VGroup()
    for x,value,label in [(-4.45,f"{f['hellaswag']:.3f}",'HellaSwag'),(-1.5,f"{f['arc_easy']:.3f}",'ARC-Easy'),
                          (1.5,f"{f['piqa']:.3f}",'PIQA'),(4.45,f"{f['winogrande']:.3f}",'WinoGrande')]:
        benchmarks.add(VGroup(text(label,19,GRAY),text(value,25,WHITE,BOLD)).arrange(DOWN,buff=.12).move_to((x,-1.68,0)))
    line2=Line((CONTENT_LEFT,-2.18,0),(CONTENT_RIGHT,-2.18,0),color=DIM,stroke_width=1)
    research=VGroup(text('POST-TRAINING STUDY',19,BLUE,BOLD).move_to((CONTENT_LEFT,-2.5,0),aligned_edge=LEFT),
                    text('Specialization measured · retention recoverable',20,WHITE).move_to((CONTENT_LEFT,-2.86,0),aligned_edge=LEFT),
                    text('Proxy gains did not transfer to GIBC',20,WHITE).move_to((CONTENT_LEFT,-3.18,0),aligned_edge=LEFT))
    selection=VGroup(text('FINAL MODEL',19,GRAY,BOLD).move_to((CONTENT_RIGHT,-2.51,0),aligned_edge=RIGHT),
                     text('BASE',46,MINT,BOLD).move_to((CONTENT_RIGHT,-2.94,0),aligned_edge=RIGHT))
    scene.add(top,primary,separators,technical,line1,results,benchmarks,line2,research,selection,status())
    return VGroup(top,primary,separators,technical,line1,results,benchmarks,line2,research,selection)

class Recap(FilmScene):
    def construct(self):
        r=recap(self);self.remove(*r);self.add(r[0],r[9])
        self.play(FadeIn(r[1]),Create(r[2]),run_time=1.3)
        self.play(FadeIn(r[3]),Create(r[4]),run_time=.8)
        self.play(FadeIn(r[5]),run_time=.8)
        self.play(FadeIn(r[6]),Create(r[7]),run_time=.9)
        self.play(FadeIn(r[8]),Indicate(r[9][1],color=MINT),run_time=1.1)
        self.wait(25-self.time)
        self.finish()
class RecapStill(FilmScene):
    def construct(self):recap(self);self.finish()
