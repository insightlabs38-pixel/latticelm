"""Independent, durable control-plane primitives for final-week research."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "post_hackathon_followup"
STATE = RESEARCH / "FOLLOWUP_CONTROLLER_STATE.json"
REGISTRY = RESEARCH / "FOLLOWUP_RUN_REGISTRY.jsonl"
DECISIONS = RESEARCH / "FOLLOWUP_DECISIONS.jsonl"
HARD_DEADLINE = "2026-10-17T20:10:00+00:00"
STATES = {"PLANNED", "IMPLEMENTING", "READY", "RUNNING", "CHECKPOINTED", "INTERRUPTED_INFRA",
          "FAILED_CODE", "FAILED_DATA", "FAILED_SCIENCE", "EVALUATING", "COMPLETED", "REJECTED", "PROMOTED", "ARCHIVED"}
_CHECKPOINT_HASH_CACHE = {}


def utc(epoch=None):
    return datetime.fromtimestamp(time.time() if epoch is None else epoch, timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def metric_rows(path):
    """Read fsynced journals without treating a torn final append as corruption."""
    data=Path(path).read_bytes();lines=data.splitlines();rows=[]
    for index,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index==len(lines)-1 and not data.endswith(b'\n'):break
            raise ValueError('Malformed committed metric record: '+str(path))
    return rows


def sync_dir(directory):
    fd = os.open(directory, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def atomic_bytes(path, content):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, path); sync_dir(path.parent)
    finally:
        Path(temp).unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())


def atomic_jsonl_add(path, row):
    """Small registry/decision logs are replaced atomically, under a writer lock."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / ("." + path.name + ".writer.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = path.read_bytes() if path.exists() else b""
        atomic_bytes(path, old + (json.dumps(row, sort_keys=True) + "\n").encode())


def decision(kind, **fields):
    atomic_jsonl_add(DECISIONS, {"at": utc(), "decision": kind, **fields})


def registrations():
    return [] if not REGISTRY.exists() else [json.loads(line) for line in REGISTRY.read_text().splitlines()]


def registered(name):
    matches = [row for row in registrations() if row["experiment_id"] == name]
    if len(matches) != 1: raise RuntimeError("expected one immutable registration for " + name)
    return matches[0]


def register(record):
    existing = [r for r in registrations() if r["experiment_id"] == record["experiment_id"]]
    if existing:
        if existing != [record]: raise RuntimeError("registration mutation refused")
        return
    atomic_jsonl_add(REGISTRY, record)


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def process_identity(pid):
    try:
        # comm may contain spaces or ')'; fields after the final ')' start at field 3.
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "boot_id": boot_id(), "start_ticks": stat[19], "process_state": stat[0]}
    except (FileNotFoundError, ProcessLookupError, PermissionError): return None


def alive(identity, expected_name):
    if not identity: return False
    actual = process_identity(identity["pid"])
    if not actual or actual["process_state"] == "Z": return False
    if any(actual[k] != identity[k] for k in ("pid", "boot_id", "start_ticks")): return False
    try: command = Path(f"/proc/{identity['pid']}/cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError): return False
    if expected_name.encode() in command:return True
    # Standalone public evaluators carry the run identity in --checkpoint,
    # including when their wrapper dies. Require the exact local script and
    # checkpoint parent; a substring match would accept unrelated processes.
    scripts={str(ROOT/'scripts'/name).encode() for name in ('evaluate_gibc.py','evaluate_wikitext103.py')}
    if not scripts.intersection(command) or b'--checkpoint' not in command:return False
    try:
        checkpoint=Path(command[command.index(b'--checkpoint')+1].decode()).resolve()
        return checkpoint.parent==(RESEARCH/'runs'/expected_name).resolve()
    except (IndexError,UnicodeDecodeError,OSError):return False


def init_state(deadline=None, reserve_hours=4):
    if STATE.exists():
        state = json.loads(STATE.read_text())
        if deadline and state["hard_deadline_utc"] != utc(parse_time(deadline)):
            raise RuntimeError("deadline differs from frozen controller state")
        return state
    epoch = parse_time(deadline or os.environ.get("LATTICELM_VM_HARD_DEADLINE_UTC", HARD_DEADLINE))
    if epoch <= time.time() + reserve_hours * 3600: raise RuntimeError("deadline already exhausted")
    state = {"schema": "followup-controller-v1", "created_utc": utc(), "local_time": datetime.now().astimezone().isoformat(),
             "hard_deadline_utc": utc(epoch), "hard_deadline_epoch": epoch,
             "scientific_stop_epoch": epoch - reserve_hours * 3600, "final_reserve_hours": reserve_hours,
             "active": None, "runs": {}, "finalized": False}
    atomic_json(STATE, state)
    decision("CAMPAIGN_INITIALIZED", hard_deadline_utc=state["hard_deadline_utc"], reserve_hours=reserve_hours)
    return state


def parse_time(value):
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if d.tzinfo is None: raise ValueError("deadline must have an explicit UTC offset")
    return d.timestamp()


def projected_seconds(tokens, rate, overhead_fraction=.08, evaluation_seconds=900):
    if tokens < 0 or rate <= 0: raise ValueError("invalid projection inputs")
    return tokens / rate * (1 + overhead_fraction) + evaluation_seconds


def failure_class(return_code, failure_receipt):
    if failure_receipt: return failure_receipt["classification"]
    if return_code is None or return_code in (-9, -15, -2): return "INTERRUPTED_INFRA"
    return "FAILED_CODE"


def verified_checkpoint(run_dir):
    """Manifest publication is the checkpoint commit point; fall back to prior."""
    path = Path(run_dir) / "checkpoint_manifest.json"
    if not path.exists(): return None
    manifest = json.loads(path.read_text())
    for item in manifest["rolling"]:
        checkpoint = Path(run_dir) / item["file"]
        if not checkpoint.is_file():continue
        stat=checkpoint.stat();key=(str(checkpoint),stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if stat.st_size != item['size_bytes']:continue
        digest=_CHECKPOINT_HASH_CACHE.get(key)
        if digest is None:
            digest=sha256(checkpoint);_CHECKPOINT_HASH_CACHE[key]=digest
        if digest == item["sha256"]:
            return item
    raise RuntimeError("no verified checkpoint in manifest")


def reconcile_rolling_manifest(run_dir):
    """Before restart, promote the verified fallback into the retained head."""
    path=Path(run_dir)/'checkpoint_manifest.json'
    if not path.exists():return
    manifest=json.loads(path.read_text());head=verified_checkpoint(run_dir)
    if manifest['rolling'][0]['file']==head['file']:return
    original=manifest['rolling'];offset=next(i for i,item in enumerate(original) if item['file']==head['file'])
    # Preserve original metadata and all weights. Only the restart selection
    # changes, so the trainer keeps an actual known-good previous checkpoint.
    atomic_json(Path(run_dir)/'CHECKPOINT_FALLBACK_RECOVERY.json',{'at':utc(),'original_rolling':original,'verified_head':head})
    manifest['rolling']=original[offset:];atomic_json(path,manifest)


def complete_evidence(run_dir, record, require_evaluation=True):
    path = Path(run_dir) / "train_receipt.json"
    if not path.exists(): return False
    r = json.loads(path.read_text())
    if r["status"] != "TRAINED" or r["logical_tokens"] != record["token_budget"]: return False
    checkpoint = verified_checkpoint(run_dir)
    if not checkpoint or checkpoint["logical_tokens"] != record["token_budget"] or checkpoint["sha256"] != r["checkpoint_sha256"]: return False
    if require_evaluation:
        ev = Path(run_dir) / "evaluation_receipt.json"
        if not ev.exists(): return False
        evaluation = json.loads(ev.read_text())
        if evaluation.get("checkpoint_sha256") != checkpoint["sha256"] or evaluation.get("status") != "COMPLETED": return False
        if record.get('schema')=='followup-run-registration-v1':
            mode=record.get('evaluation_mode','full')
            required={'latest_validation.json','evaluation_protocol.json'}
            if mode!='validation_only':required.update({'wikitext103.json','capability_samples.json',
                'capability_full.json' if mode=='full' else 'capability_screen.json'})
            if not required.issubset({i['file'] for i in evaluation.get('outputs',[])}):return False
            if evaluation.get('evaluation_mode')!=mode or not evaluation.get('evaluation_source_hashes'):return False
        if 'registration_sha256' in evaluation and evaluation['registration_sha256']!=object_hash(record):return False
        for item in evaluation.get('outputs',[]):
            output=Path(run_dir)/item['file']
            if not output.is_file():return False
            stat=output.stat();key=(str(output),stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
            if stat.st_size!=item['size_bytes']:return False
            digest=_CHECKPOINT_HASH_CACHE.get(key)
            if digest is None:digest=sha256(output);_CHECKPOINT_HASH_CACHE[key]=digest
            if digest!=item['sha256']:return False
    return True
