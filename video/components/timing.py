"""Allocate narration overruns to spare scene-ending holds, never by global speedup."""
def allocate(base,minimum,audio,limit=299.0):
    target={k:float(v) for k,v in base.items()}
    for name,seconds in audio.items():
        if seconds is not None:target[name]=max(target[name],float(seconds)+.35)
    excess=max(0,sum(target.values())-limit)
    for name in reversed(list(base)):
        if excess<=.001:break
        floor=max(minimum[name],float(audio.get(name) or 0)+.35)
        take=min(excess,max(0,target[name]-floor))
        target[name]-=take;excess-=take
    if excess>.001:raise ValueError('Narration cannot fit below runtime limit without cutting a visual beat')
    return target
