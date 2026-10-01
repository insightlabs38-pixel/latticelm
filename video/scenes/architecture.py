from manim import *
from config.theme import *
from components.layout import FilmScene,top_label,retitle,status,ambient_lines
from components.architecture import model_stack,gqa,gqa_group
from components.typography import text
from components.data import read
A=read('data/architecture.json')


def state_bars(x,y,color,heights,width=.16,gap=.12):
    out=VGroup()
    for i,h in enumerate(heights):
        out.add(RoundedRectangle(width=width,height=h,corner_radius=.025,stroke_width=0,
                                 fill_color=color,fill_opacity=.85).move_to((x+(i-(len(heights)-1)/2)*(width+gap),y-h/2,0)))
    return out


def overview(scene):
    h=top_label('01 / architecture','Where the 48.636M parameters go','MODEL PLATE')
    ambient=ambient_lines((-2.8,-1.2,.4,1.7),.055)
    stack=model_stack(-4.93,.27,1.35,2.68)
    rail=Line((-5.8,-1.1,0),(-5.8,1.62,0),color=MINT,stroke_width=2)
    blocks=text('12 Co4 blocks',29,WHITE,BOLD).move_to((-3.35,1.63,0),aligned_edge=LEFT)
    internal=VGroup(text('552 residual width',23,WHITE),text('1728 FFN width',23,WHITE),text('QK-Norm',23,MINT)).arrange(DOWN,buff=.34,aligned_edge=LEFT).move_to((-2.35,.1,0))
    divider=Line((0,-2.9,0),(0,1.9,0),color=DIM,stroke_width=1)
    qmini=VGroup(*[Circle(radius=.11,color=BLUE,fill_color=BLUE,fill_opacity=.8).move_to((-5.05+i*.52,-2.08,0)) for i in range(6)])
    qmini.add(*[Line((-5.05+i*.52,-2.18,0),(-4.54 if i<3 else -2.98,-2.48,0),color=BLUE,stroke_width=2) for i in range(6)])
    for x in (-4.54,-2.98):
        box=RoundedRectangle(width=.85,height=.37,corner_radius=.07,color=MINT,fill_color=MINT,fill_opacity=.13).move_to((x,-2.63,0))
        qmini.add(box,text('K/V',18,MINT).move_to(box))
    qmini.add(text('6Q  /  2KV real GQA',22,WHITE).move_to((-3.95,-3.03,0)))
    total=text('48,636,168',52,MINT,BOLD).move_to((.63,1.63,0),aligned_edge=LEFT)
    total_label=text('trainable parameters',22,GRAY).move_to((.66,1.03,0),aligned_edge=LEFT)
    allocation=A['parameter_allocation']
    rows=VGroup()
    spec=[('FFN matrices',allocation['ffn'],BLUE),('attention projections',allocation['attention_projections'],'#416DB8'),('embeddings + head',allocation['embeddings_and_head'],MINT)]
    for i,(label,value,color) in enumerate(spec):
        y=.42-i*.76;percent=100*value/A['parameters']
        rows.add(text(label,21,WHITE).move_to((.65,y,0),aligned_edge=LEFT))
        track=Line((3.9,y-.02,0),(4.75,y-.02,0),color=DIM,stroke_width=9)
        filled=Line((3.9,y-.02,0),(3.9+.85*percent/70.6,y-.02,0),color=color,stroke_width=9)
        rows.add(track,filled,text(f'{percent:.1f}%',22,color,BOLD).move_to((5.88,y,0),aligned_edge=RIGHT))
    rule=Line((.65,-2.05,0),(5.92,-2.05,0),color=DIM,stroke_width=1)
    details=VGroup(*[VGroup(text(v,25,WHITE,BOLD),text(k,19,GRAY)).arrange(DOWN,buff=.09) for v,k in [('256','context'),('4096','vocab'),('untied','embeddings')]]).arrange(RIGHT,buff=.78).move_to((3.25,-2.6,0))
    scene.add(ambient,h,stack,rail,blocks,internal,divider,qmini,total,total_label,rows,rule,details,status())
    return VGroup(ambient,h,stack,rail,blocks,internal,divider,qmini,total,total_label,rows,rule,details)


def co4_composition():
    # Equation terms share colors and locations with the two state lanes.
    law=MathTex(r'\operatorname{MOD}(r,c)=\operatorname{ReLU6}(',r'r^2+2r',r'+',r'c(1+|r|)',r')',font_size=48,color=WHITE).move_to((0,1.52,0))
    law[1].set_color(MINT);law[3].set_color(BLUE)
    r_label=MathTex(r'r',font_size=48,color=MINT).move_to((-5.35,.12,0))
    c_label=MathTex(r'c',font_size=48,color=BLUE).move_to((-5.35,-1.3,0))
    rvec=state_bars(-4.45,.6,MINT,[.7,1.1,.55,1.33,.87])
    cvec=state_bars(-4.45,-.83,BLUE,[1.1,.58,1.3,.76,.98])
    a=MathTex(r'r^2+2r',font_size=43,color=MINT).move_to((-1.55,.58,0))
    b=MathTex(r'c(1+|r|)',font_size=43,color=BLUE).move_to((-1.55,-.78,0))
    a_state=state_bars(-1.48,.14,MINT,[.30,.48,.35,.55,.43])
    b_state=state_bars(-1.48,-1.39,BLUE,[.55,.34,.72,.46,.58])
    arrows=VGroup(Arrow((-3.74,.14,0),(-2.75,.14,0),buff=0,color=MINT,stroke_width=3),
                  Arrow((-3.74,-1.25,0),(-2.75,-1.25,0),buff=0,color=BLUE,stroke_width=3))
    merge=VGroup(Arrow((-.5,.04,0),(1.06,-.54,0),buff=0,color=MINT,stroke_width=3),
                 Arrow((-.5,-1.25,0),(1.06,-.56,0),buff=0,color=BLUE,stroke_width=3))
    plus=MathTex('+',font_size=50,color=WHITE).move_to((1.25,-.55,0))
    clamp=VGroup(RoundedRectangle(width=1.3,height=1.55,corner_radius=.12,color=WHITE,stroke_width=2,
                                  fill_color=WHITE,fill_opacity=.035).move_to((2.85,-.55,0)),
                 text('ReLU6',25,WHITE,BOLD).move_to((2.85,-.52,0)),
                 text('bound 0–6',20,GRAY).move_to((2.85,-1.45,0)))
    output=state_bars(5.05,.05,MINT,[.72,.88,.57,1.0,.75],width=.2,gap=.11)
    output_label=text('bounded state',22,MINT).move_to((5.05,-1.6,0))
    output_arrow=Arrow((3.55,-.55,0),(4.35,-.55,0),buff=0,color=MINT,stroke_width=3)
    input_note=text('learned receptive r  +  current context c',22,GRAY).move_to((0,-2.65,0))
    return {'law':law,'inputs':VGroup(r_label,c_label,rvec,cvec),'terms':VGroup(a,b,a_state,b_state,arrows),
            'merge':VGroup(merge,plus),'clamp':clamp,'output':VGroup(output,output_label,output_arrow),'note':input_note}


def qk_composition():
    origin=np.array([-3.35,-.55,0])
    ring=Circle(radius=1.4,color=DIM,stroke_width=2).move_to(origin)
    ring2=Circle(radius=2.18,color=DIM,stroke_width=1,stroke_opacity=.45).move_to(origin)
    q=Arrow(origin,origin+np.array([2.65,1.0,0]),buff=0,color=BLUE,stroke_width=8)
    k=Arrow(origin,origin+np.array([.95,1.83,0]),buff=0,color=MINT,stroke_width=8)
    qn=Arrow(origin,origin+np.array([1.31,.49,0]),buff=0,color=BLUE,stroke_width=8)
    kn=Arrow(origin,origin+np.array([.64,1.25,0]),buff=0,color=MINT,stroke_width=8)
    labels=VGroup(text('q',27,BLUE,BOLD).move_to((-.35,.8,0)),text('k',27,MINT,BOLD).move_to((-2.35,1.38,0)))
    eq=MathTex(r'\hat q=\frac{q}{\operatorname{RMS}(q)}',r'\qquad',r'\hat k=\frac{k}{\operatorname{RMS}(k)}',font_size=44,color=WHITE).move_to((3.05,.35,0))
    note=VGroup(text('magnitude normalized',24,MINT,BOLD),text('direction preserved',23,WHITE)).arrange(DOWN,buff=.34,aligned_edge=LEFT).move_to((3.0,-1.15,0))
    attention=VGroup(Arrow((1.15,-2.25,0),(3.0,-2.25,0),buff=0,color=BLUE,stroke_width=3),text('causal attention',22,GRAY).move_to((4.25,-2.25,0)))
    return ring,ring2,q,k,qn,kn,labels,eq,note,attention


def ffn_composition():
    rails=VGroup(Line((-5.6,.75,0),(5.6,.75,0),color=DIM,stroke_width=1),Line((-5.6,-1.0,0),(5.6,-1.0,0),color=DIM,stroke_width=1))
    state=VGroup(*[Rectangle(width=.055,height=1.35,stroke_width=0,fill_color=BLUE,fill_opacity=.75).move_to((-4.4+i*.09,-.12,0)) for i in range(13)])
    wide=VGroup(*[Rectangle(width=.045,height=1.35,stroke_width=0,fill_color=MINT,fill_opacity=.65).move_to((-1.35+i*.087,-.12,0)) for i in range(32)])
    projected=state.copy().set_color(BLUE).move_to((4.5,-.12,0))
    labels=VGroup(text('552',33,WHITE,BOLD).move_to((-4.4,-1.53,0)),text('1728',33,MINT,BOLD).move_to((.0,-1.53,0)),text('552',33,WHITE,BOLD).move_to((4.5,-1.53,0)))
    captions=VGroup(text('residual state',21,GRAY).move_to((-4.4,-2.03,0)),text('SwiGLU expansion',21,GRAY).move_to((0,-2.03,0)),text('project back',21,GRAY).move_to((4.5,-2.03,0)))
    arrows=VGroup(Arrow((-3.5,-.12,0),(-2.05,-.12,0),buff=0,color=BLUE,stroke_width=3),Arrow((1.45,-.12,0),(3.6,-.12,0),buff=0,color=MINT,stroke_width=3))
    note=text('width changes inside each block; the residual stream stays 552-dimensional',22,WHITE).move_to((0,-2.75,0))
    return VGroup(rails,state,wide,projected,labels,captions,arrows,note)

class Architecture(FilmScene):
    def construct(self):
        header=top_label('01 / architecture','Where the 48.636M parameters go','MODEL GEOMETRY')
        title=text('LatticeLM',76,WHITE,BOLD).move_to((-4.4,.38,0),aligned_edge=LEFT)
        sub=text('Architecture  ·  Training  ·  Retention',25,GRAY).next_to(title,DOWN,buff=.22,aligned_edge=LEFT)
        stack=model_stack(4.4,0,2.5,4.6)
        count0=text(f"{A['parameters']:,} parameters",23,MINT).move_to((4.4,-2.72,0))
        self.add(title,sub,stack,count0)
        self.play(Transform(stack,model_stack(-3.75,-.13,1.75,3.6)),FadeOut(title),FadeOut(sub),FadeOut(count0),FadeIn(header),run_time=2.2)
        self.remove(header);self.add(*header)
        current_title,current_meta=header[1],header[2]
        constraint=MathTex(r'P<50\mathrm{M}',font_size=47,color=WHITE).move_to((1.95,1.15,0))
        result=text('48,636,168',55,MINT,BOLD).move_to((1.95,.2,0))
        insight=text('The question is how those parameters work.',25,WHITE).move_to((1.95,-.75,0))
        self.play(Write(constraint),run_time=.8)
        self.play(ReplacementTransform(constraint,result),FadeIn(insight),run_time=1)
        self.wait(2.1)
        co=co4_composition();nxt=top_label('','Inside a Co4 block','RECEPTIVE × CONTEXT')
        self.play(FadeOut(stack),FadeOut(result),FadeOut(insight),Succession(FadeOut(current_title),FadeIn(nxt[1])),Succession(FadeOut(current_meta),FadeIn(nxt[2])),Write(co['law']),FadeIn(co['inputs']),run_time=1.8)
        current_title,current_meta=nxt[1],nxt[2]
        self.play(FadeIn(co['terms']),run_time=1.8)
        self.play(Indicate(co['law'][1],color=MINT),run_time=1.5)
        self.play(Indicate(co['law'][3],color=BLUE),run_time=1.5)
        self.play(Create(co['merge']),FadeIn(co['clamp']),run_time=1.5)
        self.play(FadeIn(co['output']),FadeIn(co['note']),run_time=1.4)
        self.wait(12.8)
        diagram=gqa(y=-.05)
        title_gqa=text('three queries share each physical K/V pair',24,WHITE).move_to((0,-2.3,0))
        nxt=top_label('','6Q / 2KV grouped attention','REAL GQA')
        self.play(FadeOut(VGroup(*co.values())),Succession(FadeOut(current_title),FadeIn(nxt[1])),Succession(FadeOut(current_meta),FadeIn(nxt[2])),FadeIn(diagram[0]),run_time=1.2)
        current_title,current_meta=nxt[1],nxt[2]
        self.play(FadeIn(diagram[1]),run_time=1.2)
        self.play(Write(title_gqa),run_time=.7)
        self.play(Indicate(diagram[0][0],color=MINT),Indicate(diagram[1][0],color=MINT),run_time=1.4)
        self.wait(9.8)
        ring,ring2,q,k,qn,kn,labels,eq,note,attention=qk_composition()
        nxt=top_label('','Normalizing attention geometry','QK-NORM')
        self.play(FadeOut(diagram),FadeOut(title_gqa),Succession(FadeOut(current_title),FadeIn(nxt[1])),Succession(FadeOut(current_meta),FadeIn(nxt[2])),Create(ring),Create(ring2),GrowArrow(q),GrowArrow(k),run_time=1.4)
        current_title,current_meta=nxt[1],nxt[2]
        self.play(Transform(q,qn),Transform(k,kn),FadeIn(eq),run_time=1.8)
        self.play(FadeIn(note),FadeIn(attention),run_time=.8)
        self.wait(8.1)
        flow=ffn_composition()
        nxt=top_label('','The 552-dimensional residual stream','FFN EXPANSION')
        self.play(FadeOut(VGroup(ring,ring2,q,k,eq,note,attention)),Succession(FadeOut(current_title),FadeIn(nxt[1])),Succession(FadeOut(current_meta),FadeIn(nxt[2])),FadeIn(flow[0]),FadeIn(flow[1]),FadeIn(flow[4][0]),run_time=1.1)
        current_title,current_meta=nxt[1],nxt[2]
        self.play(TransformFromCopy(flow[1],flow[2]),FadeIn(flow[4][1]),FadeIn(flow[6][0]),run_time=1.5)
        self.play(TransformFromCopy(flow[2],flow[3]),FadeIn(flow[4][2]),FadeIn(flow[6][1]),run_time=1.5)
        self.play(FadeIn(flow[5]),FadeIn(flow[7]),run_time=.8)
        self.wait(7.4)
        plate=overview(self);self.remove(*plate)
        self.play(FadeOut(flow),FadeOut(current_title),FadeOut(current_meta),FadeOut(header[0]),FadeOut(header[3]),FadeIn(plate),run_time=1.2)
        if self.time>87.8:raise RuntimeError(f'Architecture exceeds target: {self.time:.2f}s')
        self.wait(88-self.time)
        self.finish()

class ArchitectureStill(FilmScene):
    def construct(self):overview(self);self.finish()
