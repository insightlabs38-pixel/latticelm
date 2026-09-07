from datetime import datetime,timedelta,timezone
import scripts.run_seed_replication_master as master

def test_replication_classification():
 stable={**master.CONTROL,"wikitext_ppl":61,"wikitext_bpb":1.90,"data_d_validation_loss":3.5}
 high={**master.CONTROL,"wikitext_ppl":90,"wikitext_bpb":2.5,"data_d_validation_loss":4.2}
 assert master.replication_class(stable)=="STABLE"
 assert master.replication_class(high)=="HIGH VARIANCE"

def test_deadline_gate_reserves_evaluation():
 now=datetime.now(timezone.utc);state={"soft_deadline":(now+timedelta(hours=12.5)).isoformat(),"hard_deadline":(now+timedelta(hours=14)).isoformat(),"observed_throughput":5003}
 assert master.enough_time(state)
 state["hard_deadline"]=(now+timedelta(hours=2)).isoformat()
 assert not master.enough_time(state)

def test_uncertainty_declares_unpaired_limitation():
 candidate={**master.CONTROL,"hellaswag":.28};candidate.pop("raw_gibc")
 rows=master.uncertainty(master.CONTROL,candidate,"x")
 assert len(rows)==4
 assert all("raw paired predictions unavailable" in row["method"] for row in rows)
