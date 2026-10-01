import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_all_scene_previews():
    paths=json.loads((ROOT/'renders/preview/scene_paths.json').read_text())
    assert len(paths)==7
    assert all(Path(x).exists() for x in paths.values())
