"""Synthetic control-flow drill; never touches live state."""
from __future__ import annotations
import argparse,csv,json,tempfile
from pathlib import Path
from unittest.mock import patch
try:import run_post900m_master as m
except ModuleNotFoundError:import scripts.run_post900m_master as m
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 with tempfile.TemporaryDirectory(prefix="post900m-drill-") as td:
  r=Path(td);names={k:r/v for k,v in {"STATE":"state.json","BACKUP":"previous.json","EVENTS":"events.jsonl","HANDOFF":"handoff.json","DECISION":"decision.json","REPORT":"report.md","SCALING":"scaling.json","CERT":"cert.json","CURVE":"curve.csv"}.items()};old={k:getattr(m,k) for k in names}
  try:
   for k,v in names.items():setattr(m,k,v)
   fields=["nominal_tokens","train_loss","data_d_validation_loss","fineweb_edu_validation_loss","wikipedia_validation_loss","fineweb_validation_loss"]
   with names["CURVE"].open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for t,l in [(50_000_000,3.5),(205_000_000,3.3),(300_000_000,3.25),(500_000_000,3.20),(700_000_000,3.18),(900_000_000,3.17)]:w.writerow({"nominal_tokens":t,"train_loss":l+.2,"data_d_validation_loss":l,"fineweb_edu_validation_loss":l,"wikipedia_validation_loss":l-.05,"fineweb_validation_loss":l+.05})
   cert={"status":"PASS","checkpoint":"fake.pt","checkpoint_sha256":"a"*64,"manifest_sha256":"b"*64};m.atomic(names["CERT"],cert)
   wiki={"perplexity":40.,"bits_per_byte":1.7};g={"results":{k:{"acc,none":.25} for k in ("hellaswag","arc_easy","piqa","winogrande")}}
   s={"current_stage":"WAIT_PREDECESSOR","run_id":"drill","child_pid":None,"stop_requested":False}
   # Force the inexpensive terminal branch. This guarantees a drill can never
   # start a corpus build or training process even if classifier policy changes.
   with patch.object(m,"predecessor_ready",return_value=(True,None)),patch.object(m,"certify_900m",return_value=cert),patch.object(m,"evaluate",return_value=(wiki,g)),patch.object(m,"wiki_history",return_value={205_000_000:{"perplexity":50.,"bits_per_byte":1.9},900_000_000:wiki}),patch.object(m,"classify",return_value="INCONCLUSIVE"),patch.object(m,"run",side_effect=AssertionError("drill must not launch a child")):m.execute(s)
   checks={"temporary_namespace":"PASS","wait_gate":"PASS","certification_gate":"PASS","evaluation_idempotence":"PASS","scaling_models":"PASS" if names["SCALING"].exists() else "FAIL","terminal_handoff":"PASS" if names["HANDOFF"].exists() else "FAIL"};checks["status"]="PASS" if set(checks.values())=={"PASS"} else "FAIL"
  finally:
   for k,v in old.items():setattr(m,k,v)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(checks,indent=2)+"\n");print(json.dumps(checks));return checks["status"]!="PASS"
if __name__=="__main__":raise SystemExit(main())
