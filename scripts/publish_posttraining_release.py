#!/usr/bin/env python3
import argparse,json,os
from pathlib import Path
from huggingface_hub import get_token
from latticelm.hf_storage import upload_checkpoint
def main():
 p=argparse.ArgumentParser();p.add_argument("--directory",type=Path,required=True);p.add_argument("--repo",default=os.environ.get("LATTICELM_HF_REPO","insightlabs38-pixel/LatticeLM-research"));p.add_argument("--path",default="final-posttraining/selected");p.add_argument("--output",type=Path,required=True);a=p.parse_args();token=get_token()
 if not token:raise RuntimeError("Hugging Face credential is unavailable")
 revision=upload_checkpoint(a.directory,a.repo,a.path,token);result={"repo_id":a.repo,"path":a.path,"revision":revision,"verified":True};a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
if __name__=="__main__":main()
