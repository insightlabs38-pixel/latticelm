from manim import *
from config.theme import *
from components.layout import FilmScene,top_label,status,ambient_lines
from components.typography import text
from components.data import read
A=read('data/architecture.json');P=read('data/pretraining.json')


def pipeline(scene):
    h=top_label('02 / pretraining','DATA-D-v4: a certified training stream','DATA PIPELINE')
    x0=-5.8; length=11.6; cursor=x0
    mix=VGroup();shares=sorted(P['data_sources'].items(),key=lambda kv:-kv[1])
    names={'fineweb_edu':'FineWeb-Edu','wikipedia':'Wikipedia','fineweb':'FineWeb','finemath':'FineMath'}
    colors=[BLUE,'#527BBE','#4268A7',MINT]
    for (key,share),color in zip(shares,colors):
        w=length*share;cx=cursor+w/2
        mix.add(Rectangle(width=w,height=.4,stroke_width=0,fill_color=color,fill_opacity=.85).move_to((cx,1.18,0)))
        mix.add(text(f'{names[key]}  {share*100:.1f}%',19,WHITE if color!=MINT else MINT).move_to((cx,2.02 if key=='finemath' else 1.67,0)))
        cursor+=w
    source=text('SOURCE MIX',20,GRAY,BOLD).move_to((0,.6,0))
    stages=VGroup();xs=[-4.55,-1.35,1.75,4.85]
    names=['normalize','dedup + ID','tokenize','DATA-D-v4']
    for i,(x,name) in enumerate(zip(xs,names)):
        color=MINT if i==3 else BLUE
        circle=Circle(radius=.46,color=color,stroke_width=2.2,fill_color=color,fill_opacity=.09).move_to((x,-.23,0))
        stages.add(circle,text(str(i+1).zfill(2),20,color,BOLD).move_to(circle),text(name,23,WHITE if i<3 else MINT,BOLD).move_to((x,-1.04,0)))
        if i:stages.add(Arrow((xs[i-1]+.65,-.23,0),(x-.65,-.23,0),buff=0,color=GRAY,stroke_width=2))
    streams=VGroup(Arrow((-4.55,.96,0),(-4.55,.3,0),buff=0,color=BLUE,stroke_width=2,stroke_opacity=.7))
    total=text(f"{A['dataset_tokens']:,}",52,MINT,BOLD).move_to((0,-2.14,0))
    label=text('CERTIFIED USABLE TOKENS',21,GRAY,BOLD).move_to((0,-2.76,0))
    scene.add(h,mix,source,stages,streams,total,label,status())
    return VGroup(h,mix,source,stages,streams,total,label)


def optimizer(scene):
    h=top_label('02 / pretraining','Parameter geometry sets the optimizer','MUON + ADAMW')
    # Shape glyphs link parameter types to the optimizer assignment.
    left=VGroup();right=VGroup()
    for row in range(4):
        for col in range(6):
            left.add(Square(side_length=.26,stroke_color=BLUE,stroke_width=1.2,fill_color=BLUE,fill_opacity=.35).move_to((-4.13+col*.35,1.15-row*.35,0)))
    for x in (2.8,3.27,3.74):
        for j in range(5):
            right.add(RoundedRectangle(width=.27,height=.2,corner_radius=.03,stroke_color=MINT,stroke_width=1,
                                       fill_color=MINT,fill_opacity=.3).move_to((x,1.21-j*.27,0)))
    left.add(text('MATRIX PARAMETERS',20,BLUE,BOLD).move_to((-3.28,1.82,0)))
    right.add(text('VECTORS · EMBEDDINGS · NORMS',20,MINT,BOLD).move_to((3.25,1.82,0)))
    route=VGroup(Arrow((-3.15,-.35,0),(-3.15,-.85,0),buff=0,color=BLUE,stroke_width=2.5),
                 Arrow((3.25,-.35,0),(3.25,-.85,0),buff=0,color=MINT,stroke_width=2.5))
    numbers=VGroup()
    for x,value,method,detail,color in [(-3.2,f"{A['muon_parameters']:,}",'Muon','90.65%',BLUE),(3.22,f"{A['adamw_parameters']:,}",'AdamW','9.35%',MINT)]:
        numbers.add(text(method,24,color,BOLD).move_to((x,-1.08,0)),text(value,37,WHITE,BOLD).move_to((x,-1.57,0)),text(detail,23,color,BOLD).move_to((x,-2.03,0)))
    start=-5.62;end=5.62;split=start+(end-start)*A['muon_parameters']/A['parameters']
    bar=VGroup(Rectangle(width=split-start,height=.25,stroke_width=0,fill_color=BLUE,fill_opacity=.85).move_to(((start+split)/2,-2.78,0)),
               Rectangle(width=end-split,height=.25,stroke_width=0,fill_color=MINT,fill_opacity=.85).move_to(((split+end)/2,-2.78,0)))
    scene.add(h,left,right,route,numbers,bar,status())
    return VGroup(h,left,right,route,numbers,bar)


def schedule(scene):
    h=top_label('02 / pretraining','Late WSD decay, visible validation gain','LEARNING-RATE SCHEDULE')
    x0,x1=-5.65,5.65;w=x1-x0;warm=x0+.02*w;decay=x0+.85*w;early=x0+(1.5/1.9)*w
    cooldown=Rectangle(width=x1-decay,height=4.35,stroke_width=0,fill_color=MINT,fill_opacity=.085).move_to(((decay+x1)/2,-.16,0))
    rule=VGroup(Line((x0,.12,0),(x1,.12,0),color=DIM,stroke_width=1.5),Line((x0,-2.2,0),(x1,-2.2,0),color=DIM,stroke_width=1.5),
                Line((decay,2.0,0),(decay,-2.2,0),color=MINT,stroke_width=1,stroke_opacity=.45))
    def lr_y(x):
        if x<warm:return .42+1.02*(x-x0)/(warm-x0)
        if x<decay:return 1.44
        t=(x-decay)/(x1-decay);return .42+1.02*(1+np.cos(np.pi*t))/2
    lr=ParametricFunction(lambda t:np.array([x0+t*w,lr_y(x0+t*w),0]),t_range=[0,1],color=BLUE,stroke_width=5)
    top_labels=VGroup(text('2% warmup',21,BLUE).move_to((-4.45,1.89,0)),
                      text('83% stable',21,WHITE).move_to((-.15,1.89,0)),
                      text('15% cosine',21,MINT).move_to((4.65,1.89,0)))
    val_label=text('VALIDATION LOSS',20,GRAY,BOLD).move_to((x0,-.55,0),aligned_edge=LEFT)
    pts=VGroup(Dot((early,-1.1,0),radius=.115,color=WHITE),Dot((x1,-1.88,0),radius=.13,color=MINT))
    trend=Arrow((early+.12,-1.14,0),(x1-.12,-1.84,0),buff=0,color=MINT,stroke_width=4)
    vals=VGroup(text('1.5B',22,WHITE,BOLD).move_to((early,-.78,0)),
                text('1.9B',22,MINT,BOLD).move_to((x1-.38,-1.37,0)))
    result=VGroup(text('2.9145',43,WHITE,BOLD),text('→',35,GRAY),text('2.6305',43,MINT,BOLD)).arrange(RIGHT,buff=.22).move_to((-2.48,-1.45,0))
    result_caption=text('late validation improvement',21,GRAY).move_to((-2.5,-2.65,0))
    tokens=VGroup(text('0',18,GRAY).move_to((x0,-3.04,0)),text('1.9B tokens',18,GRAY).move_to((x1-.3,-3.04,0)))
    scene.add(h,cooldown,rule,lr,top_labels,val_label,pts,trend,vals,result,result_caption,tokens,status())
    return VGroup(h,cooldown,rule,lr,top_labels,val_label,pts,trend,vals,result,result_caption,tokens)

class Training(FilmScene):
    def construct(self):
        p=pipeline(self)
        self.remove(p[0]);self.add(*p[0])
        self.wait(16.6)
        o=optimizer(self);self.remove(*o)
        self.play(Succession(FadeOut(VGroup(*p[1:])),FadeIn(VGroup(*o[1:5]))),FadeIn(o[5]),Succession(FadeOut(p[0][1]),FadeIn(o[0][1])),Succession(FadeOut(p[0][2]),FadeIn(o[0][2])),run_time=1.1)
        self.wait(15.6)
        s=schedule(self);self.remove(*s)
        self.play(Succession(FadeOut(VGroup(*o[1:])),FadeIn(VGroup(*[s[i] for i in range(1,len(s)) if i!=3]))),FadeIn(s[3]),Succession(FadeOut(o[0][1]),FadeIn(s[0][1])),Succession(FadeOut(o[0][2]),FadeIn(s[0][2])),run_time=1.1)
        # The shared vertical cursor links the LR path and both measured checkpoints.
        x0,x1=-5.65,5.65;early=x0+(1.5/1.9)*(x1-x0)
        tracker=ValueTracker(x0)
        cursor=always_redraw(lambda:Line((tracker.get_value(),1.53,0),(tracker.get_value(),-2.22,0),color=WHITE,stroke_width=1.5,stroke_opacity=.65))
        self.add(cursor)
        self.play(tracker.animate.set_value(early),run_time=8,rate_func=linear)
        self.play(Indicate(s[6][0],color=WHITE),run_time=1)
        self.play(tracker.animate.set_value(x1),run_time=7,rate_func=linear)
        self.play(Indicate(s[6][1],color=MINT),Indicate(s[9],color=MINT),run_time=1.5)
        if self.time>74.8:raise RuntimeError(f'Training exceeds target: {self.time:.2f}s')
        self.wait(75-self.time)
        self.finish()

class PipelineStill(FilmScene):
    def construct(self):pipeline(self);self.finish()
class OptimizerStill(FilmScene):
    def construct(self):optimizer(self);self.finish()
class ScheduleStill(FilmScene):
    def construct(self):schedule(self);self.finish()
