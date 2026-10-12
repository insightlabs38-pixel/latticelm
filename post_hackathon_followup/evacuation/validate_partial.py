"""Read-only exact-resume validation; never executes a training step."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from followup_common import registered, object_hash, sha256, atomic_json
from followup_train import build_model, sampler, batch_hash
from train_final_production import load_arrays
from latticelm.config import LatticeConfig
from latticelm.final_recipe import make_optimizer, wsd_lr_scale

run = Path('post_hackathon_followup/runs/F8-CO4-FULLCONTROL-50M-S1337')
entry = json.loads((run / 'checkpoint_manifest.json').read_text())['rolling'][0]
path = run / entry['file']
assert sha256(path) == entry['sha256']
ck = torch.load(path, map_location='cpu', weights_only=False)
record = registered(ck['experiment_id'])
cfg = LatticeConfig(**record['config'])
torch.set_num_threads(2)
assert ck['registration_sha256'] == object_hash(record)
assert ck['config_sha256'] == object_hash(record['config'])
# The native checkpoint stores tuples; JSON registration stores arrays.
# Use the trainer's exact comparison rule.
assert json.loads(json.dumps(ck['config'])) == record['config']
for relative, digest in ck['source_hashes'].items():
    assert sha256(Path(relative)) == digest, relative
assert sha256(Path(record['manifest_path'])) == ck['manifest_sha256']
assert sha256(Path(record['tokenizer_path'])) == ck['tokenizer_sha256']
model = build_model(record)
model.load_state_dict(ck['model'], strict=True)
optimizer, partition = make_optimizer(model, cfg.optimizer, cfg.learning_rate,
    cfg.weight_decay, (cfg.adam_beta1, cfg.adam_beta2))
assert partition == ck['optimizer_partition']
optimizer.load_state_dict(ck['optimizers'])
top, arrays, valid = load_arrays(Path(record['manifest_path']))
mixture = sampler(record, cfg, top, arrays)
mixture.load_state_dict(ck['mixture'])
mixture.cycle = tuple(ck['sampling_cycle'])
mixture.annealed = ck['sampling_annealed']
for name, values in ck['sampler_diagnostics'].items():
    for key, value in values.items():
        setattr(mixture.streams[name], key, value)
random.setstate(ck['python_rng'])
np.random.set_state(ck['numpy_rng'])
torch.set_rng_state(ck['torch_rng'])
assert batch_hash(mixture) == ck['next_batch_sha256']
assert ck['processed_tokens'] == ck['step'] * cfg.batch_size * cfg.context_length
fraction = ck['step'] / ck['schedule_total_steps']
region = 'warmup' if fraction < cfg.warmup_fraction else 'stable' if fraction < cfg.warmup_fraction + cfg.stable_fraction else 'decay'
receipt = {
    'status': 'TRUNCATED_RESOURCE_EXHAUSTION', 'experiment_id': ck['experiment_id'],
    'actual_tokens': ck['logical_tokens'], 'target_tokens': record['token_budget'],
    'schedule_tokens': record.get('schedule_token_budget', record['token_budget']),
    'schedule': 'fresh WSD', 'step': ck['step'], 'wsd_region': region,
    'current_lrs': [g['lr'] for g in optimizer.param_groups],
    'wsd_lr_scale': wsd_lr_scale(ck['step'], ck['schedule_total_steps'], cfg.warmup_fraction, cfg.stable_fraction, cfg.decay_fraction),
    'checkpoint_path': str(path), 'checkpoint_sha256': entry['sha256'],
    'checkpoint_source_git_sha': ck['source_commit'], 'config_hash': ck['config_sha256'],
    'dataset_manifest_hash': ck['manifest_sha256'], 'tokenizer_hash': ck['tokenizer_sha256'],
    'wall_seconds': ck['wall_seconds'], 'measured_tokens_per_second': ck['logical_tokens'] / ck['wall_seconds'],
    'reason': 'GCP credit exhaustion / emergency VM evacuation',
    'strict_load_resume_validation': 'PASS',
    'checks': ['checkpoint hash', 'model strict load', 'optimizer state load', 'RNG restore', 'sampler restore', 'next-batch identity', 'token accounting', 'registered config/source/data/tokenizer identity'],
    'original_train_receipt_preserved': True,
    'dense_300m_run': 'already completed on 2026-10-11; not truncated',
    'initial_validation_probe': 'Python tuple/list equality rejected native-vs-JSON config; canonical JSON equality and all hashes match; comparison corrected to trainer semantics',
}
atomic_json(run / 'evacuation_receipt.json', receipt)
atomic_json(run.parents[1] / 'evacuation/PARTIAL_RUN_RECEIPT.json', receipt)
print(json.dumps(receipt, indent=2))
