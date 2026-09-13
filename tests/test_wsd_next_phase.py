from scripts.wsd_next_phase import TASKS, classify, pareto_frontier

def ep(bpb, score, valid=True, offsets=None):
    x={"valid":valid,"wikitext_bpb":bpb,**{t:score for t in TASKS}}
    if offsets:
        for k,v in offsets.items(): x[k]+=v
    return x

def test_all_classification_branches():
    assert classify([ep(1.70,.30),ep(1.67,.305),ep(1.63,.32)])=="LATE_HORIZON_CLEARLY_BEST"
    assert classify([ep(1.70,.30),ep(1.64,.32),ep(1.68,.30)])=="INTERMEDIATE_HORIZON_CLEARLY_BEST"
    assert classify([ep(1.64,.32),ep(1.68,.30),ep(1.70,.30)])=="EARLY_HORIZON_CLEARLY_BEST"
    assert classify([ep(1.650,.30),ep(1.645,.301),ep(1.638,.302)])=="WSD_EFFECTIVE_BUT_SCALING_SATURATING"
    assert classify([ep(1.650,.300),ep(1.649,.301),ep(1.648,.302)])=="HORIZONS_EFFECTIVELY_TIED"
    assert classify([ep(1.65,.30),ep(1.64,.30),ep(1.63,.30)],False)=="WSD_INSUFFICIENT"
    mixed=[ep(1.66,.30),ep(1.63,.30,offsets={"arc_easy":.02}),ep(1.61,.30,offsets={"piqa":.02,"winogrande":-.02})]
    assert classify(mixed)=="PARETO_INCONCLUSIVE"
    assert classify([ep(1.6,.3,False),ep(1.6,.3),ep(1.6,.3)])=="BLOCKED_OR_INVALID"

def test_only_clear_late_can_imply_direct_scaling():
    outcomes={classify([ep(1.70,.30),ep(1.67,.305),ep(1.63,.32)])}
    assert outcomes=={"LATE_HORIZON_CLEARLY_BEST"}

def test_cheapest_means_cheapest_on_actual_pareto_frontier():
    branches=[{"label":"early","tokens":1,"metrics":ep(1.7,.30)},
              {"label":"middle","tokens":2,"metrics":ep(1.6,.31)},
              {"label":"late","tokens":3,"metrics":ep(1.5,.31,offsets={"arc_easy":-.02})}]
    assert [x["label"] for x in pareto_frontier(branches)] == ["middle","late"]
