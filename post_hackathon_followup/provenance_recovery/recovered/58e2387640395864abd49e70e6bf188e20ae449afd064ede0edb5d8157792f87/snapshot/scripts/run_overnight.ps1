$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
python -m latticelm.train --config configs/dense_128.json --experiment dense_128 --download --tokenizer artifacts/tokenizers/tinyshakespeare_100k_4k.json
python -m latticelm.train --config configs/lattice_64k_128.json --experiment lattice_64k_128 --download --tokenizer artifacts/tokenizers/tinyshakespeare_100k_4k.json
foreach ($threads in 1, 2, 4, 8) { python -m latticelm.benchmark --threads $threads }
