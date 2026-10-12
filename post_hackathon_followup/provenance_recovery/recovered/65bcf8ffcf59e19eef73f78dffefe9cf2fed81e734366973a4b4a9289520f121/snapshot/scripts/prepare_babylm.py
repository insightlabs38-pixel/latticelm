"""Download and deterministically split the official BabyLM 2026 Strict-Small corpus."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "BabyLM-community/BabyLM-2026-Strict-Small"
REVISION = "c92ab16b4f08858304b0815706065b3354d8fc0a"
FILES = (
    "bnc_spoken.train.txt", "childes.train.txt", "gutenberg.train.txt",
    "open_subtitles.train.txt", "simple_wiki.train.txt", "switchboard.train.txt",
)


def main() -> None:
    target = ROOT / "artifacts/data/babylm_2026_strict_small"
    target.mkdir(parents=True, exist_ok=True)
    sources = []
    train_parts: list[str] = []
    validation_parts: list[str] = []
    for name in FILES:
        path = target / name
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}?download=true"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        # The official repository exposes training data only. Take the final 2%
        # of every source as a fixed architecture-search validation split.
        cut = max(1, int(len(lines) * 0.98))
        train_parts.append("".join(lines[:cut]))
        validation_parts.append("".join(lines[cut:]))
        sources.append({"file": name, "url": url, "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(), "lines": len(lines),
                        "train_lines": cut, "validation_lines": len(lines) - cut})
    train_text, validation_text = "\n".join(train_parts), "\n".join(validation_parts)
    (target / "train.txt").write_text(train_text, encoding="utf-8")
    (target / "validation.txt").write_text(validation_text, encoding="utf-8")
    manifest = {
        "dataset": REPO, "revision": REVISION, "official_use": "training corpus",
        "split_policy": "last 2% of lines within each source held out; no benchmark evaluation data",
        "train_words_whitespace": len(train_text.split()),
        "validation_words_whitespace": len(validation_text.split()),
        "train_bytes_utf8": len(train_text.encode()),
        "validation_bytes_utf8": len(validation_text.encode()),
        "sources": sources,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "artifacts/dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
