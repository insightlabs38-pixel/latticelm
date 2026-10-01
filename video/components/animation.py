from manim import *

def reveal(scene,group,run_time=.8):
    scene.play(LaggedStart(*[Create(x) if isinstance(x,VMobject) and not isinstance(x,Text) else Write(x) for x in group],lag_ratio=.08),run_time=run_time)
