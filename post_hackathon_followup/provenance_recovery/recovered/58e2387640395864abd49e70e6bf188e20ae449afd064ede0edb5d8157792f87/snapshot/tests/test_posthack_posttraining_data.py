from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from latticelm.config import LatticeConfig
from latticelm.posthack_variants import build_posthack_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import posthack_posttrain_data as data_module
import posthack_posttrain_eval as eval_module
import posthack_posttrain as train_module


def test_posttraining_early_batches_cover_every_registered_resource(tmp_path):
    source = tmp_path / "aligned.jsonl"
    sources = ("commonsense_qa", "sciq", "qasc")
    with source.open("w") as handle:
        for name in sources:
            for index in range(100):
                handle.write(json.dumps({"resource": name, "prompt": f"{name} {index}",
                                         "answer": "yes"}) + "\n")
    first = train_module.ordered_rows(source, 915123, "seeded_resource_round_robin_v1")
    repeated = train_module.ordered_rows(source, 915123, "seeded_resource_round_robin_v1")
    assert first == repeated
    assert {row["resource"] for row in first[:16]} == set(sources)
    assert {row["resource"] for row in first} == set(sources)
    assert first != train_module.ordered_rows(source, 915124, "seeded_resource_round_robin_v1")
    assert [row["resource"] for row in train_module.ordered_rows(source, 1, "sequential")[:16]] == [sources[0]] * 16


def test_loader_excludes_answers_that_erase_the_prompt(tmp_path):
    tokenizer = Tokenizer.from_file(str(train_module.TOKENIZER))
    source = tmp_path / "examples.jsonl"
    source.write_text("".join(json.dumps({"resource": "piqa", "prompt": "Question:",
                                          "answer": answer}) + "\n"
                              for answer in ("short", "very long answer " * 1000)))
    examples, skipped = train_module.load_examples(
        source, tokenizer, 256, 915123, "seeded_resource_round_robin_v1")
    assert len(examples) == 1
    assert skipped == 1


def test_posttraining_files_are_distinct_and_skill_validation_is_disjoint(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "RESEARCH", tmp_path)
    monkeypatch.setattr(data_module, "OUT", tmp_path / "posttraining/data")
    monkeypatch.setattr(data_module, "prompt_answer",
                        lambda name, row, labeled=True: (row["prompt"], row["answer"]))

    def rows(name, split):
        if split == "train":
            items = [{"prompt": f"{name} train", "answer": "yes"}]
            if name == "hellaswag":
                items.append({"prompt": "competition heldout", "answer": "no"})
        elif name in {"sciq", "commonsense_qa", "qasc"}:
            items = [{"prompt": f"{name} train", "answer": "duplicate"},
                     {"prompt": f"{name} skill validation", "answer": "yes"}]
        else:
            items = [{"prompt": "competition heldout", "answer": "yes"}]
        return items, {"resource": name, "split": split}

    monkeypatch.setattr(data_module, "parquet_rows", rows)
    report = data_module.build()
    folder = tmp_path / "posttraining/data"
    read = lambda name: [json.loads(line) for line in (folder / name).read_text().splitlines()]
    generic = read("generic_train.jsonl")
    aligned = read("aligned_train.jsonl")
    skills = read("heldout_skill_validation.jsonl")
    task = read("task_train.jsonl")
    assert {row["resource"] for row in generic} == {"sciq", "qasc"}
    assert {row["resource"] for row in aligned} == {"sciq", "qasc", "commonsense_qa"}
    assert {row["resource"] for row in skills} == {"sciq", "qasc", "commonsense_qa"}
    assert {row["prompt_sha256"] for row in skills}.isdisjoint(
        {row["prompt_sha256"] for row in generic + aligned + task})
    assert all(row["prompt"] != "competition heldout" for row in task)
    assert len(report["files"]) == 4


def test_heldout_skill_evaluator_loads_checkpoint_and_scores_by_resource(tmp_path, monkeypatch):
    cfg = LatticeConfig(vocab_size=4096, d_model=32, ffn_hidden=64,
                        n_heads=4, n_kv_heads=2, n_layers=2,
                        context_length=16, architecture="co4_causal",
                        real_gqa=True, qk_norm=True)
    model = build_posthack_model(cfg)
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(),
                "variant": "baseline"}, checkpoint)
    data = tmp_path / "skills.jsonl"
    data.write_text("".join(json.dumps({"resource": name, "prompt": "Question: sky? Answer:",
                                        "answer": "blue"}) + "\n"
                            for name in ("sciq", "commonsense_qa", "qasc")))
    monkeypatch.setattr(eval_module, "DATA", data)
    output = tmp_path / "result.json"
    result = eval_module.evaluate(checkpoint, output)
    assert set(result["by_resource"]) == {"sciq", "commonsense_qa", "qasc"}
    assert all(row["completion_tokens"] > 0 for row in result["by_resource"].values())
    assert math.isfinite(result["aggregate_nll_nats_per_completion_token"])
    assert json.loads(output.read_text())["checkpoint_sha256"] == result["checkpoint_sha256"]
