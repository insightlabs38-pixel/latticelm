import json
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def test_manifest():
    manifest=json.loads((ROOT/'screenshots/manifest.json').read_text())
    assert len(manifest)==7
    assert len(set(x['filename'] for x in manifest))==7
    for x in manifest:
        assert Image.open(ROOT/x['filename']).size==(1920,1080)
        assert x['status'] in ('provisional','final')
        if 'high_resolution_filename' in x:
            assert Image.open(ROOT/x['high_resolution_filename']).size==(2560,1440)
