from manim import *
from config.theme import *
from components.layout import FilmScene,top_label,status
from components.architecture import model_stack
from components.typography import text,number
from components.plots import smooth_path
from components.data import read
A=read('data/architecture.json');P=read('data/pretraining.json')
class TrainingSetupStill(FilmScene):
    def construct(self):
        h=top_label('07 / system','Architecture and optimization as one system','TRAINING SETUP')
        stack=model_stack(-3.6,-.15,1.8,3.8)
        source=text(f"DATA-D-v4  /  {A['dataset_tokens']/1e9:.4f}B certified tokens",23,WHITE).move_to((-3.35,-2.45,0))
        muon=number(f"{100*A['muon_parameters']/A['parameters']:.2f}%",'Muon  /  2D matrices',BLUE,41).move_to((1.75,1.05,0))
        adam=number(f"{100*A['adamw_parameters']/A['parameters']:.2f}%",'AdamW  /  vectors + embeddings',MINT,41).move_to((1.75,-.42,0))
        arrows=VGroup(Arrow((-2.55,.7,0),(-.15,1.05,0),color=BLUE),Arrow((-2.55,-.9,0),(-.15,-.4,0),color=MINT))
        sched=smooth_path([(.1,-2.25),(1,-1.8),(3.3,-1.8),(5.3,-2.35)],BLUE,3)
        lbl=text(f"WSD  /  {P['wsd']['warmup']*100:g}% warmup · {P['wsd']['stable']*100:g}% stable · {P['wsd']['decay']*100:g}% decay",19,GRAY).move_to((2.9,-2.72,0))
        self.add(h,stack,source,muon,adam,arrows,sched,lbl,status());self.finish()
