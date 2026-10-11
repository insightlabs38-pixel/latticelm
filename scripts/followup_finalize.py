"""Evidence-driven reporting and restartable, verified deadline closeout."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from followup_common import ROOT,RESEARCH,STATE,atomic_json,atomic_bytes,complete_evidence,decision,registrations,utc,metric_rows


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def evidence_text(filename):
    d=read(RESEARCH/'decisions'/filename)
    if d is None:return 'Unresolved: paired endpoint evaluation is not yet complete.'
    e=d.get('evidence')
    if isinstance(e,dict) and 'paired_loss' in e:
        return (f"Candidate `{e['candidate']}` against `{e['control']}`: loss delta "
          f"{e['paired_loss']['candidate_minus_control']:.6g}, 95% paired interval {e['paired_loss']['ci95']}; "
          f"BPB delta {e['paired_bpb']['candidate_minus_control']:.6g}, interval {e['paired_bpb']['ci95']}; "
          f"clear task regression {e['capability']['any_clear_regression']}; wall ratio {e['wall_ratio']:.4g}. "
          f"Gate {'positive' if d.get('run') else 'closed/not promoted'}. See `decisions/{filename}` for source-wise evidence and limits.")
    return f"{'Positive' if d.get('run') else 'Not confirmed/promoted'}; detailed evidence in `decisions/{filename}`."


def reports(state,plots=False):
    summaries=[]
    for r in registrations():
        if r.get('smoke'):continue
        run=RESEARCH/'runs'/r['experiment_id'];receipt=read(run/'train_receipt.json',{})
        validation=read(run/'latest_validation.json',{});wiki=read(run/'wikitext103.json',{})
        evaluation=read(run/'evaluation_receipt.json',{})
        status=state.get('runs',{}).get(r['experiment_id'],{}).get('status','PLANNED')
        complete=complete_evidence(run,r) if receipt.get('status')=='TRAINED' else False
        n=receipt.get('processed_tokens',0);cfg=r['config'];stages=r.get('full_co4',{}).get('block_types',[]).count('modulated') if r.get('full_co4') else 0
        summaries.append({'experiment_id':r['experiment_id'],'status':'COMPLETED' if complete else status,
          'completed_evidence':complete,'role':r.get('role',r.get('hypothesis')),'seed':cfg['seed'],'parameters':r['parameter_count'],
          'logical_tokens':receipt.get('logical_tokens',state.get('runs',{}).get(r['experiment_id'],{}).get('live_tokens',0)),
          'processed_tokens':n,'wall_seconds':receipt.get('wall_seconds'),'tokens_per_second':receipt.get('logical_tokens',0)/max(receipt.get('wall_seconds',1),1e-9) if receipt else None,
          'peak_rss_bytes':receipt.get('peak_rss_bytes'),'depth_total_blocks':cfg['n_layers'],'co4_stages':stages,
          'heldout_loss':validation.get('frozen_mixture_weighted_loss'),'source_losses':validation.get('source_losses'),
          'bits_per_byte':wiki.get('bits_per_byte'),'perplexity':wiki.get('perplexity'),'capability_scope':evaluation.get('capability_scope'),
          'capability':read(run/('capability_full.json' if r.get('evaluation_mode','full')=='full' else 'capability_screen.json'),{}).get('results'),
          'matrix_training_flops_proxy_6NT':6*r['parameter_count']*n,
          'additional_co4_latent_matrix_flops_proxy':6*stages*n*(3*cfg['d_model']**2+2*cfg['d_model']*cfg['ffn_hidden']),
          'additional_co4_element_operation_proxy':3*stages*n*(12*(cfg['context_length']+1)/2*cfg['d_model']+6*13**2*cfg['d_model']),
          'compute_caveat':'6NT is a coarse matrix-training proxy including embeddings. Prefix-local latent QKV/MLP matrix applications add explicit extra work using the same weights; triadic arithmetic/selected attention add a separate element-operation proxy. These are not measured FLOPs. Top-k comparisons, memory traffic, framework overhead and standard attention costs are not fully counted. Wall time is authoritative.',
          'checkpoint_sha256':receipt.get('checkpoint_sha256'),'source_commit':r['code_commit'],'source_hashes':r['source_hashes'],
          'config':cfg,'full_co4':r.get('full_co4')})
        summary=summaries[-1]
        lines=['# '+r['experiment_id'],'','Post-submission research. Updated '+utc()+'.','',
          'Status: '+summary['status']+'. '+('Endpoint checkpoint and evaluation evidence verified.' if complete else 'Incomplete; no final scientific conclusion.'),'',
          'Role: '+str(summary['role']),
          f"Seed: {cfg['seed']}; parameters: {r['parameter_count']:,}; token target: {r['token_budget']:,}.",
          'Source commit: `'+r['code_commit']+'`.',
          'Checkpoint SHA256: `'+str(summary['checkpoint_sha256'])+'`.',
          'Initialization/schedule: '+r.get('initialization_policy','See immutable registration.')+'.',
          'Full-Co4: '+json.dumps(r.get('full_co4'))+'.',
          '','','| Metric | Value |','|---|---|']
        for key in ('logical_tokens','processed_tokens','wall_seconds','tokens_per_second','peak_rss_bytes','heldout_loss','bits_per_byte','perplexity','capability_scope'):
            lines.append('| '+key+' | '+str(summary[key])+' |')
        lines.extend(['','Source losses: '+json.dumps(summary['source_losses'])+'.',
          'Capability results: '+json.dumps(summary['capability'])+'.',
          '','Compute/depth claims require the paired comparison reports and their uncertainty; parameter matching does not imply compute matching.',
          'Full-Co4 diagnostic statistics, when available, are in `architecture_diagnostics.json`. Configuration and source hashes are in the immutable run registry and `FOLLOWUP_RUN_SUMMARY.json`.'])
        atomic_bytes(run/'REPORT.md',('\n'.join(lines)+'\n').encode())
    atomic_json(RESEARCH/'FOLLOWUP_RUN_SUMMARY.json',{'at':utc(),'post_submission':True,'runs':summaries})
    plan=read(RESEARCH/'FOLLOWUP_EXPERIMENT_PLAN.json',{})
    family_decisions={p.name:read(p) for p in (RESEARCH/'decisions').glob('*.json')}
    atomic_json(RESEARCH/'FOLLOWUP_SCIENTIFIC_DECISIONS.json',{'at':utc(),'decisions':family_decisions,'policy':plan.get('promotion_policy')})
    causal=[read(p) for p in (RESEARCH/'runs').glob('*/causality_reference_receipt.json')]
    causal_status='Passed all '+str(len(causal))+' tested production configurations.' if causal and all(c['passed'] for c in causal) else 'Tiny oracle/author/causality tests implemented; configuration-specific C0 certificates are pending or failed. Consult certificates and C0 receipts.'
    git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    archive=read(RESEARCH/'last_archive_receipt.json',{})
    confirmed=read(RESEARCH/'decisions/full-confirmation.json',{})
    winner=read(RESEARCH/'decisions/full-winner.json',{})
    answers=[
      ('At 32M/300M, does existing MOD-Co4 differ meaningfully from dense?',evidence_text('dense-300m-closure.json')+' This is a separately registered fresh dense300M comparison. Historical dense100M had ended its WSD; it was not silently continued.'),
      ('Does D4 annealing reproduce?',evidence_text('d4-replication.json')+' The sampler cycle, source draw counts and transition receipt preserve the exact late mixture. BPB-only movement is distinguished from coherent recipe promotion.'),
      ('Does Differential-Co4 reproduce?',evidence_text('differential-replication.json')+' One fresh matched seed7331 pair supplements the historical pilot; no automatic four-seed tournament.'),
      ('Which exact full-Co4 architectures were implemented?','IHMS author implementation pinned to e25e9119173d59fdb9e4548ef640734dbd19f067: feature-wise independent Q/K/V top-k, latent-modulated triadic Q/K/V and scatter readout. B is pure modulated; C is modulated+MHSA. Faithful BabyLM A remains unavailable under the user-confirmed source limitation.'),
      ('Faithful, adapted or hybridized?','The released IHMS operator is independently fixture-tested. Its autoregressive LM adaptation is explicitly adapted: causal prefix-local selection/readout, learned token/position embeddings, untied LM head, widths/depths, initialization and inherited optimizer. C additionally hybridizes with causal MHSA. Neither is claimed to faithfully reproduce the unavailable BabyLM code.'),
      ('Did causal leak tests pass?',causal_status+' Tests cover multiple seeds/prefix lengths, changed future tokens, selection indices and future embedding gradients; forward, gradients and one optimizer step match an independent literal oracle.'),
      ('Did full Co4 beat shallow dense, current MOD or deeper dense?','Comparator-specific paired loss/BPB/task evidence appears in C2 promotion and winner decisions. '+('Winner: family '+str(winner.get('family'))+'.' if winner else 'No evaluated winner selected yet.')+' A depth tie against deep dense does not establish an advantage over parameter-matched shallow dense; attribute those axes separately.'),
      ('Per token, parameter, FLOP proxy, wall time or depth?','FOLLOWUP_RUN_SUMMARY.json reports each axis independently. Learning curves record quality vs tokens and wall time; total parameters, stages/blocks, peak RSS, 6NT and extra triadic work are disclosed. A model has no efficiency claim from parameter count alone.'),
      ('Did the winning formation reproduce across two seeds?',evidence_text('full-confirmation.json')+' Two primary seeds maximum:1337 and2026. Confirmation requires consistent direction on the same comparator/axis and rejects clear task regressions; item confidence intervals do not measure training-seed uncertainty.'),
      ('What hyperparameters were architecture-specific and why?','At most four C1 settings per family: inherited LR/init, LR×0.5, LR×2, or initσ0.01. Extra arms open only for unstable/non-learning or clearly worse validation. Selection uses held-out loss, never downstream scores. Each selected setting is frozen before fresh C2/C3 runs; see C1 selection decisions.'),
      ('Does the published large depth/sample claim hold in causal LM?','Supported only for the matched comparisons and replicated axes documented above; otherwise unresolved or not reproduced here. Quadratic causal prefix processing does not verify published near-linear scaling. A negative result does not disprove the source vision/RL architecture.'),
      ('Which architecture deserves GPU-scale work?',('IHMS family '+str(confirmed.get('family'))+' is a candidate on its replicated axes; inspect comparison-specific resource tradeoffs before scaling.' if confirmed.get('run') else 'No full-Co4 architecture qualifies without coherent two-seed evidence. Differential/D4 are candidates only if their own replication gates pass.')),
      ('Which lines should stop?','Failed or non-promoted compatibility/screen/replication branches stop at their frozen gates. Previously deferred PHM, Clifford, MoE, CGLS, value residual, Engram, post-training, DeltaNet, BLT, Titans and multi-token work remain deferred. No claim that an unrun conditional experiment failed.'),
      ('Are all irreplaceable artifacts safely remote?','Verified coverage is determined by FOLLOWUP_ARCHIVE_MANIFEST.json and HF_REPRESENTATIVE_LOAD_RECEIPT.json; upload exit alone is insufficient. Historical remote duplicates are reused, canonical shards/manifests are mirrored as missing delta, and raw document indexes can be reconstructed from verified gzip plus original source SHA. '+('Closeout remote verification completed.' if state.get('finalized') else 'Migration/closeout is ongoing; complete coverage is not yet claimed.')),
      ('What exact Git/HF state defines the research?','Reporting source revision `'+git+'`; branch `posthack-followup-2026-10`, frozen main/submission `43832b4c7bfe8be007ca6b9d1042aa2041c7f280`. HF `insightlabs38-pixel/LatticeLM-research`, prefix `post-hackathon-followup/2026-10-11`, last verified receipt '+str(archive.get('revision','pending'))+'. Final closeout commit/revision are recorded in FOLLOWUP_CLOSEOUT_RECEIPT.json and the archive manifest.')]
    lines=['# Follow-up final synthesis','','Post-submission research. Updated '+utc()+'.',
      'Status: '+('FINALIZED' if state.get('finalized') else 'LIVE — incomplete endpoints are explicitly unresolved.'),'']
    for index,(q,a) in enumerate(answers,1):lines.extend([f'## {index}. {q}','',a,''])
    atomic_bytes(RESEARCH/'FOLLOWUP_FINAL_SYNTHESIS.md',('\n'.join(lines).rstrip()+'\n').encode())
    matrix=['# Follow-up execution matrix','','| Experiment | Status | Parameters | Tokens | Loss | BPB | Wall hours |','|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        wall='pending' if r['wall_seconds'] is None else f"{r['wall_seconds']/3600:.3f}"
        matrix.append(f"| {r['experiment_id']} | {r['status']} | {r['parameters']:,} | {r['logical_tokens']:,} | {r['heldout_loss']} | {r['bits_per_byte']} | {wall} |")
    atomic_bytes(RESEARCH/'FOLLOWUP_EXPERIMENT_MATRIX.md',('\n'.join(matrix)+'\n').encode())
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        directory=RESEARCH/'plots';directory.mkdir(exist_ok=True)
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for r in summaries:
            path=RESEARCH/'runs'/r['experiment_id']/'validation_history.jsonl'
            if not path.exists():path=path.parent/'historical_learning_curve.jsonl'
            if not path.exists():continue
            history=metric_rows(path)
            if not history:continue
            ys=[x['frozen_mixture_weighted_loss'] for x in history]
            axes[0].plot([x['logical_tokens']/1e6 for x in history],ys,label=r['experiment_id'])
            # Each validation entry points to the exact manifest checkpoint whose
            # elapsed wall time was stored; never substitute zero for missing time.
            mp=read(path.parent/'checkpoint_manifest.json',{'rolling':[],'milestones':[]})
            times={x['sha256']:x['wall_seconds'] for x in mp['rolling']+mp['milestones']}
            matched=[x for x in history if x.get('checkpoint_sha256') in times]
            if matched:axes[1].plot([times[x['checkpoint_sha256']]/3600 for x in matched],[x['frozen_mixture_weighted_loss'] for x in matched],label=r['experiment_id'])
        for ax,label in zip(axes,['Logical tokens (M)','Elapsed training wall hours (checkpoint receipts)']):ax.set_xlabel(label);ax.set_ylabel('Frozen-mixture held-out loss');ax.grid(alpha=.2)
        if summaries:axes[0].legend(fontsize=5,loc='best')
        fig.tight_layout();fig.savefig(directory/'learning_curves.png',dpi=160);plt.close(fig)
    return summaries


def publish_git():
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!='posthack-followup-2026-10':raise RuntimeError('Closeout refuses unexpected branch')
    paths=[str(p.relative_to(ROOT)) for p in (RESEARCH).rglob('*') if p.is_file() and
      p.suffix in ('.json','.jsonl','.md','.png','.csv') and not p.name.startswith('.') and
      p.name not in ('FOLLOWUP_CONTROLLER_STATE.json','FOLLOWUP_DECISIONS.jsonl','FOLLOWUP_ARCHIVE_MANIFEST.json','last_archive_receipt.json','FOLLOWUP_CLOSEOUT_RECEIPT.json','FOLLOWUP_LIVE_SCHEDULE.md') and
      not any(s in p.relative_to(RESEARCH).parts for s in ('runs','sources'))]
    paths += [str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('followup_*.py')]
    paths += ['scripts/posthack_followup_master.py','scripts/evaluate_gibc.py','scripts/evaluate_wikitext103.py','src/latticelm/full_co4.py','src/latticelm/full_co4_reference.py','tests/test_full_co4.py','tests/test_followup_controller.py']
    paths += [str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_followup_*.py')]
    paths += [str(p.relative_to(ROOT)) for p in (ROOT/'tests/fixtures/ihms').glob('*') if p.is_file()]
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    # The run namespace is deliberately ignored to keep weights out of Git.
    # Add only these generated, small Markdown reports with an exact allowlist.
    run_reports=[str(p.relative_to(ROOT)) for p in (RESEARCH/'runs').glob('*/REPORT.md') if not p.parent.name.startswith('SMOKE-')]
    if run_reports:subprocess.run(['git','add','-f','--',*run_reports],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m','Finalize post-submission follow-up research evidence'],cwd=ROOT,check=True)
    subprocess.run(['git','-c','credential.helper=','-c','credential.helper=/home/insightlabs38/.config/latticelm-followup/git-credential','push','https://github.com/insightlabs38-pixel/latticelm','HEAD:posthack-followup-2026-10'],cwd=ROOT,check=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    main=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip()
    if main!='43832b4c7bfe8be007ca6b9d1042aa2041c7f280':raise RuntimeError('Frozen main changed')
    return revision


def finalize(state):
    if state.get('active'):raise RuntimeError('Closeout requires stopped scientific workers')
    for r in registrations():
        if r.get('smoke'):continue
        info=state['runs'].setdefault(r['experiment_id'],{})
        if info.get('status','PLANNED') in ('PLANNED','READY','INTERRUPTED_INFRA') and not complete_evidence(RESEARCH/'runs'/r['experiment_id'],r,require_evaluation=False):
            info.update(status='REJECTED',reason='Deadline left insufficient time; unrun/incomplete, not a negative scientific result.')
    reports(state,plots=True)
    atomic_json(RESEARCH/'FINAL_ARCHIVE_REQUEST.json',{'at':utc(),'include_partial_resume_checkpoints':True})
    from followup_archive import archive,representative_loads
    # Lock makes this idempotent across an interrupted finalization or uploader.
    import fcntl
    with (RESEARCH/'.archive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        archive(final=True);loads=representative_loads(require_new=True)
        closing=dict(state,finalized=True)
        try:
            reports(closing,plots=True)
            revision=publish_git()
            atomic_json(RESEARCH/'FOLLOWUP_CLOSEOUT_RECEIPT.json',{'at':utc(),'final_git_commit':revision,'main_unchanged':True,
                 'remote_loads_passed':loads['passed'],'scientific_results_complete':all(complete_evidence(RESEARCH/'runs'/r['experiment_id'],r) for r in registrations() if r['experiment_id'] in ('F32-DENSE-S1337-FRESH300M','F16-D4-S2026-FRESH150M','F8-MOD-S7331-FRESH100M','F8-DIFF-S7331-FRESH100M')) and any(complete_evidence(RESEARCH/'runs'/r['experiment_id'],r) for r in registrations() if r.get('full_co4') and 'modulated' in r['full_co4']['block_types'] and r['token_budget']>=50_000_000),
                 'conditional_run_semantics':'Closed gates and deadline skips are reported as unrun, never negative experiments.'})
            manifest=archive(final=True)
        except Exception:
            reports(state,plots=True)
            raise
        state.update(finalized=True,finalized_git_commit=revision,finalized_utc=utc());atomic_json(STATE,state)
        decision('FOLLOWUP_FINALIZED',git_commit=revision,verified_file_count=len(manifest['files']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports-only',action='store_true');args=p.parse_args();state=read(STATE,{})
    if args.reports_only:reports(state)
    else:finalize(state)
