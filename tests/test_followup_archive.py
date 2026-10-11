"""Small archive protocol tests: batching, verified bytes, persistent cooldown."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import followup_archive as archive
from followup_common import atomic_json


def test_batched_commit_verifies_all_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(archive,'ROOT',tmp_path);monkeypatch.setattr(archive,'RESEARCH',tmp_path/'research')
    archive.RESEARCH.mkdir();secret=tmp_path/'token';secret.write_text('fixture');monkeypatch.setattr(archive,'SECRET',secret)
    attrs=tmp_path/'attrs';attrs.write_bytes(b'*.int32 filter=lfs diff=lfs merge=lfs -text\n')
    files=[]
    for i in range(4):
        p=archive.RESEARCH/f'{i}.json';p.write_text(json.dumps({'value':i}));files.append(p)
    class API:
        def __init__(self):self.calls=[];self.remote={}
        def repo_info(self,*_,**__):return SimpleNamespace(private=True,sha='initial',siblings=[])
        def create_commit(self,repo,operations,**kwargs):
            operations=list(operations);self.calls.append(operations)
            assert kwargs['num_threads']==1
            for op in operations:
                p=Path(op.path_or_fileobj)
                self.remote[op.path_in_repo]=SimpleNamespace(rfilename=op.path_in_repo,size=p.stat().st_size,lfs=None,blob_id=archive.git_blob_hash(p))
            return SimpleNamespace(oid='revision'+str(len(self.calls)))
        def get_paths_info(self,repo,paths,**kwargs):return [self.remote[p] for p in paths]
    api=API()
    with patch.object(archive,'HfApi',return_value=api),patch.object(archive,'hf_hub_download',return_value=str(attrs)),patch.object(archive,'source_snapshot'),patch.object(archive,'selected_files',return_value=files),patch.object(archive,'historical_files',return_value=[]),patch.object(archive.os,'nice'),patch.object(archive.subprocess,'check_output',return_value='gitrevision'),patch.object(archive,'decision'):
        manifest=archive.archive()
    assert len(api.calls)==2 and len(api.calls[0])==4
    assert len(manifest['files'])==4 and all(e['verified'] for e in manifest['files'].values())


def test_cooldown_prevents_any_new_network_or_hashing_work(tmp_path,monkeypatch):
    monkeypatch.setattr(archive,'RESEARCH',tmp_path)
    atomic_json(tmp_path/'HF_ARCHIVE_RETRY.json',{'retry_epoch':9e12,'status':'RATE_LIMIT_COOLDOWN'})
    with patch.object(archive,'HfApi',side_effect=AssertionError('network must not run')),patch.object(archive,'historical_files',side_effect=AssertionError('expensive hashing must not run')):
        assert archive.archive()['deferred']


def test_archive_includes_both_training_and_validation_shards(tmp_path,monkeypatch):
    monkeypatch.setattr(archive,'ROOT',tmp_path)
    monkeypatch.setattr(archive,'cached_hash',archive.sha256)
    base=tmp_path/'artifacts/data/data_d_v4/canonical-2250m';base.mkdir(parents=True)
    top={'shards':[],'validation_shards':[]}
    expected=[]
    for kind in top:
        shard=base/(kind+'.int32');shard.write_bytes(b'\0'*16)
        child=base/(kind+'.json')
        atomic_json(child,{'path':shard.name,'sha256':archive.sha256(shard)})
        top[kind].append({'manifest_path':child.name,'manifest_sha256':archive.sha256(child)})
        expected.extend((child,shard))
    atomic_json(base/'manifest.json',top)
    assert all(p in archive.historical_files() for p in expected)
