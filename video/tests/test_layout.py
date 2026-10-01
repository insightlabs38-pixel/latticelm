import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manim import Rectangle
from components.layout import QARegistry

def test_safe_frame_and_collision():
    q=QARegistry();q.add('a',Rectangle(width=1,height=1).move_to((0,0,0)));q.add('b',Rectangle(width=1,height=1).move_to((.2,0,0)))
    assert any('overlaps' in x for x in q.problems())
    q=QARegistry();q.add('outside',Rectangle(width=1,height=1).move_to((6.5,0,0)))
    assert any('outside safe frame' in x for x in q.problems())
