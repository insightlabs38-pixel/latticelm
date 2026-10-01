SCENES=[('Opening',8),('Architecture',88),('Training',75),('PostTraining',45),('Recovery',45),('FinalProof',12),('Recap',25)]
TARGET_SECONDS=sum(s for _,s in SCENES)
MAX_SECONDS=300.0
FPS=60
assert TARGET_SECONDS==298
