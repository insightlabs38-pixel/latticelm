"""Version-specific historical evidence; never infer missing trainer provenance."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import torch

from followup_common import ROOT, RESEARCH, atomic_json, sha256, utc
from latticelm.final_recipe import wsd_lr_scale


def verify_historical_control(name, checkpoint, digest):
    """Accept only the source-recoverable, independently audited control family.

    The early A8 runs need their missing source snapshots or fresh replacements;
    a matching endpoint/config alone does not establish their trainer provenance.
    """
    if name not in ('R32-CO4-CONTROL', 'R16-S2026-CONTROL'):
        raise ValueError('Historical trainer source is not recoverable: ' + name)
    audit = json.loads((RESEARCH / 'R32_CO4_PROVENANCE_AUDIT.json').read_text())
    if not audit.get('passed'):
        raise ValueError('Source-family provenance audit has not passed')
    for rel, expected in audit['source_hashes'].items():
        if sha256(ROOT / rel) != expected:
            raise ValueError('Audited source family changed: ' + rel)
    trainer = audit['source_hashes']['scripts/posthack_train.py']
    if checkpoint.get('trainer_sha256') != trainer:
        raise ValueError('Historical checkpoint trainer differs from audited source family')
    if name == 'R32-CO4-CONTROL':
        if digest != audit['checkpoint_sha256']:
            raise ValueError('R32 endpoint differs from its provenance audit')
        return audit
    path = RESEARCH / 'R16_CONTROL_PROVENANCE_AUDIT.json'
    if path.exists():
        result = json.loads(path.read_text())
        if result.get('passed') and result['checkpoint_sha256'] == digest:
            # Pin immutable source receipts, not only a previously written PASS.
            for rel, expected in result['evidence_hashes'].items():
                if sha256(ROOT / rel) != expected:
                    raise ValueError('Historical evidence changed: ' + rel)
            return result
    old = ROOT / 'post_hackathon_research'
    record = next(r for r in map(json.loads, (old / 'RUN_REGISTRY.jsonl').read_text().splitlines())
                  if r['experiment_id'] == name)
    run = old / 'runs' / name
    receipt = json.loads((run / 'result.json').read_text())
    events = [e for e in map(json.loads, (old / 'controller_events.jsonl').read_text().splitlines())
              if e.get('experiment_id') == name]
    counts = Counter(e['event'] for e in events)
    cfg = record['config']; target = record['token_budget']
    quantum = cfg['batch_size'] * cfg['context_length']; steps = math.ceil(target / quantum)

    def check(condition, message):
        if not condition:
            raise ValueError('R16 historical provenance: ' + message)

    check(receipt['checkpoint_sha256'] == digest == (run / 'latest.sha256').read_text().strip(), 'endpoint hash')
    check(checkpoint['logical_tokens'] == checkpoint['target_tokens'] == target == 150_000_000, 'token target')
    check(checkpoint['step'] == receipt['steps'] == steps, 'endpoint step')
    check(json.loads(json.dumps(checkpoint['config'])) == cfg, 'registered config')
    check(counts['TRAIN_START'] == counts['TRAIN_END'] == 1 and counts['RESUMED'] == 0, 'fresh uninterrupted event history')
    start = next(e for e in events if e['event'] == 'TRAIN_START')
    check(start['trainer_sha256'] == trainer, 'start trainer identity')
    check(start['target_tokens'] == target, 'start target')
    check(checkpoint['optimizer_partition'] == record['optimizer_partition'], 'optimizer partition')
    check(checkpoint['manifest_sha256'] == record['sampling_manifest_sha256'] ==
          sha256(ROOT / record['manifest_path']), 'data manifest identity')
    check(checkpoint['tokenizer_sha256'] == record['tokenizer_sha256'] ==
          sha256(ROOT / record['tokenizer_path']), 'tokenizer identity')
    # mmap only; all token shards were already hash-verified for the R32 audit
    # and again at the active dense launch against this same manifest.
    import numpy as np
    from latticelm.data_d import SourceStream
    from train_final_production import WeightedMixture, next_batch_hash
    manifest = ROOT / record['manifest_path']; top = json.loads(manifest.read_text())
    arrays = {source: [] for source in top['mixture_definition']}
    for item in top['shards']:
        child = manifest.parent / item['manifest_path']; meta = json.loads(child.read_text())
        check(sha256(child) == item['manifest_sha256'], 'child manifest identity')
        shard = manifest.parent / meta['path']
        check(shard.stat().st_size == meta['token_count'] * 4, 'shard size')
        arrays[meta['source']].append(np.memmap(shard, mode='r', dtype='<i4'))
    streams = {source: SourceStream(arrays[source], cfg['context_length'], cfg['seed'] + 17 * i)
               for i, source in enumerate(arrays)}
    mixture = WeightedMixture(streams, top['mixture_definition'], cfg['batch_size'])
    check(tuple(checkpoint['sampling_cycle']) == mixture.cycle and not checkpoint['sampling_annealed'], 'sampling cycle')
    draws = Counter(); previous = 0; maximum_lr_error = 0.
    for row in map(json.loads, (run / 'metrics.jsonl').read_text().splitlines()):
        step = row['step']; check(step == previous + 1, 'contiguous steps'); previous = step
        lr = cfg['learning_rate'] * wsd_lr_scale(step, steps, cfg['warmup_fraction'], cfg['stable_fraction'], cfg['decay_fraction'])
        maximum_lr_error = max(maximum_lr_error, abs(row['lr'] - lr))
        check(row['logical_tokens'] == min(step * quantum, target) and row['processed_tokens'] == step * quantum, 'token accounting')
        offset = (step - 1) * cfg['batch_size']
        check(tuple(row['source_labels']) == tuple(mixture.cycle[(offset + i) % len(mixture.cycle)] for i in range(cfg['batch_size'])), 'source order')
        draws.update(row['source_labels'])
    check(previous == steps and maximum_lr_error <= 1e-15, 'whole-run WSD trajectory')
    check(checkpoint['mixture'] == {'counter': steps * cfg['batch_size'], 'draws': dict(draws)}, 'sampler cursors')
    mixture.load_state_dict(checkpoint['mixture'])
    check(next_batch_hash(mixture) == checkpoint['next_batch_sha256'], 'next batch replay')
    evidence = [run / 'result.json', run / 'metrics.jsonl', run / 'latest.sha256', old / 'RUN_REGISTRY.jsonl', old / 'controller_events.jsonl']
    result = {'schema': 'historical-control-provenance-v1', 'at': utc(), 'passed': True,
              'control': name, 'checkpoint_sha256': digest, 'trainer_sha256': trainer,
              'source_hashes': audit['source_hashes'], 'event_counts': dict(counts),
              'total_steps': steps, 'maximum_lr_error': maximum_lr_error,
              'saved_next_batch_sha256': checkpoint['next_batch_sha256'],
              'evidence_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in evidence},
              'evidence_limit': 'No initial R16 checkpoint archived. Fresh initialization follows the recovered trainer source and single start/no resume history; no new forward replay was run. Token shards inherit prior R32/dense launch hash verification.'}
    atomic_json(path, result)
    return result
