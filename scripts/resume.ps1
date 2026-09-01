param([Parameter(Mandatory=$true)][string]$Checkpoint, [Parameter(Mandatory=$true)][string]$Config, [Parameter(Mandatory=$true)][string]$Experiment)
$env:PYTHONPATH = 'src'
python -m latticelm.train --config $Config --experiment $Experiment --resume $Checkpoint
