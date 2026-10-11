"""Frozen, bounded conditional experiment templates for the final-week campaign."""
from __future__ import annotations
import copy
import json
from followup_common import ROOT,RESEARCH,atomic_json,utc


def full_count(cfg,order):
    d=cfg['d_model'];h=cfg['ffn_hidden'];m=order.count('modulated');s=order.count('standard')
    return (2*cfg['vocab_size']+cfg['context_length']+2+bool(m))*d+m*(3*d*d+2*d*h+h+8*d)+s*(4*d*d+2*d*h+h+9*d)


def profile(base,order,target):
    cfg=copy.deepcopy(base);cfg.update(n_layers=len(order),n_heads=4,n_kv_heads=4,real_gqa=False,qk_norm=False,tie_embeddings=False)
    width=min(range(64,1025,4),key=lambda d:abs(full_count({**cfg,'d_model':d,'ffn_hidden':4*d},order)-target))
    cfg.update(d_model=width,ffn_hidden=4*width)
    return {'config':cfg,'full_co4':{'family':'ihms_hybrid' if 'standard' in order and 'modulated' in order else 'ihms_causal',
            'block_types':order,'top_k':12,'chunk_size':8,'initialization_scale':.02,'latent_count':1,'modulated_heads':1},
            'parameter_count_analytical':full_count(cfg,order)}


def shallow_control(recipe):
    cfg=copy.deepcopy(recipe['config']);order=['standard']*cfg['n_layers'];target=recipe['parameter_count_analytical']
    cfg['ffn_hidden']=min(range(1,5*cfg['d_model']+1),key=lambda h:abs(full_count({**cfg,'ffn_hidden':h},order)-target))
    full=copy.deepcopy(recipe['full_co4']);full.update(family='ihms_causal',block_types=order,differential=False)
    return {'config':cfg,'full_co4':full,'parameter_count_analytical':full_count(cfg,order),
            'matching_policy':'Same width, positional embeddings, depth, tokenizer; FFN width adjusted for total parameter match.'}


def prepare():
    path=RESEARCH/'FOLLOWUP_EXPERIMENT_PLAN.json'
    if path.exists():return json.loads(path.read_text())
    historical={r['experiment_id']:r for r in map(json.loads,(ROOT/'post_hackathon_research/RUN_REGISTRY.jsonl').read_text().splitlines())}
    base=historical['A8-CO4-S1337']['config']
    families={}
    for name,order in [('B',['modulated']*2),('C',['modulated','modulated','standard'])]:
        main=profile(base,order,8_000_000)
        ladders={}
        for count in (1,2,4):
            structure=['modulated']*count if name=='B' else (['modulated','standard'] if count==1 else ['modulated','modulated','standard']*(count//2))
            candidate=profile(base,structure,8_000_000)
            ladders[str(count)]={'candidate':candidate,'shallow_dense':shallow_control(candidate)}
        scale=profile(base,order,16_000_000)
        families[name]={'main':main,'shallow_dense':shallow_control(main),'depth_ladder':ladders,
          'scale_16m':{'candidate':scale,'shallow_dense':shallow_control(scale)},
          'tuning_arms':[{'key':'BASE','global_lr_multiplier':1.,'initialization_scale':.02},
                         {'key':'LR-HALF','global_lr_multiplier':.5,'initialization_scale':.02},
                         {'key':'LR-DOUBLE','global_lr_multiplier':2.,'initialization_scale':.02},
                         {'key':'INIT-HALF','global_lr_multiplier':1.,'initialization_scale':.01}]}
    plan={'schema':'followup-experiment-plan-v1','created_utc':utc(),'post_submission':True,
      'unavailable_family_A':'Faithful BabyLM implementation unavailable; user forbids invented fidelity. No paper reconstruction is required for the source-grounded branch.',
      'source_pin':'e25e9119173d59fdb9e4548ef640734dbd19f067','families':families,
      'stage_budgets':{'compatibility':5_000_000,'screen':50_000_000,'primary_confirmation':100_000_000,'extension_total':200_000_000,'scale16':150_000_000},
      'confirmation_seeds':[1337,2026],'max_primary_full_co4_seeds_per_winning_formation':2,
      'max_compatibility_arms_per_family':4,'max_100m_promoted_families':2,
      'fresh_stage_policy':'C1, C2 and C3 are separate fresh runs with each endpoint horizon registered before training. C1 tuning never uses downstream scores. No zero-LR checkpoint is silently extended.',
      'extension_policy':'Optional 100M->200M follow-up is a separately registered paired extension phase for both winning seeds and controls: preserve model/optimizer/RNG/sampler; disclose one new 100M WSD phase and LR restart. Never claim uninterrupted 200M training.',
      'promotion_policy':{'confirmed_by_user':True,'quality':'Coherent held-out loss and BPB evidence; paired uncertainty; reject statistically clear task regressions.',
       'other_axes':'Sample, depth, compute and wall-clock judged separately. Favorable sample/depth efficiency may promote a quality tie. No rigid percentage thresholds.',
       'final_claim':'Winning formation must remain directionally consistent across exactly two primary seeds.',
       'uncertainty':'Stratified paired DATA-D validation-block bootstrap, paired WikiText raw-row bootstrap, and paired exact capability-item tests with four-task multiplicity correction; item uncertainty is not training-seed uncertainty.'},
      'optional_experiments':{'depth1':'Prepare/run for a confirmed positive family with matched shallow control.',
        'depth2':'Reuse the main confirmation configuration when it already has two modulated stages; do not duplicate it.',
        'depth4':'Run only for a confirmed family whose late validation curve still improves and only if time permits.',
        'scale16_seed1':'One 16M/150M source-family run and matched shallow control after two consistent 8M seeds.',
        'scale16_seed2':'One additional 16M seed only if the first confirms direction and fits the deadline.',
        'differential16':'One fresh 16M/150M matched MOD/Differential pair only after Differential replication.',
        'full_differential':'One 8M/50M screen only after independent Differential replication and two-seed full-Co4 confirmation.'},
      'fairness':'All runs use the frozen tokenizer/data, 32x256 batches and float32. Exact measured parameter counts and source-specific extra operations are reported. Parameter matching is approximate where width is discrete; no claim of equal compute from equal parameters.',
      'source_adaptations':['Autoregressive prefix-local feature-wise top-k and current-token scatter readout; one latent/head per modulated block.',
        'Prefix-local latent evolves between stages and is reset per batch; no persistent mu dynamics.',
        'Learned token/position embeddings and untied LM head replace patch/classification readout.',
        'GELU MLP expansion four and LayerNorm/residuals preserved; block depth/width adapted to 8M/16M.',
        'Borrowed common Muon/AdamW recipe and normal .02 initialization are disclosed LM adaptations, not a released LM recipe. Bounded tuning tests only three further predefined settings.',
        'A hybrid uses two modulated blocks plus one ordinary causal MHSA block; depth ladder reports both Co4 stages and total blocks.',
        'Causal prefix processing is quadratic; no claim of published near-linear scaling.']}
    atomic_json(path,plan)
    atomic_json(RESEARCH/'FOLLOWUP_PREFLIGHT_QUEUE.json',{'experiment_ids':['F16-D4-S2026-FRESH150M','F8-MOD-S7331-FRESH100M','F8-DIFF-S7331-FRESH100M']})
    return plan

if __name__=='__main__':prepare();print('Prepared all source-grounded and conditional experiment templates.')
