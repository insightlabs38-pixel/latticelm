# LatticeLM video and figures

This directory contains the Manim source, visual components, metric inputs, narration timing, VHS tapes, and build utilities for the LatticeLM submission film. The source is organized as seven scenes: Opening, Architecture, Training, PostTraining, Recovery, FinalProof, and Recap. The final metric store identifies BASE as the selected checkpoint and keeps the 256-example reasoning proxy separate from the official pretraining metrics.

The repository retains the seven final 1920×1080 submission stills in [`final/screenshots/`](final/screenshots/). The full composed MP4, scene renders, contact sheets, intermediate frames, 1440p variants, local render tools, and delivery ZIPs are generated outputs and are excluded from Git. The last documented no-audio preview was 1920×1080 at 60 fps and about 298 seconds; it is not a narrated final film.

## Requirements

Use the project environment and install `video/requirements.txt`. Rendering also requires Manim's system dependencies (Cairo/Pango and LaTeX) and FFmpeg. VHS terminal captures require the VHS tool and its dependencies; VHS tapes are retained as source. Audio is not included. Narration text and timing are in [`final/narration/`](final/narration/); provide recorded scene clips separately to compose a narrated export.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install -r video/requirements.txt
```

## Build

Run from the repository root. `render_all.py` saves scene renders under the ignored `video/renders/` directory. The screenshot generator writes review captures under ignored paths and updates `video/screenshots/manifest.json`; the checked-in submission stills are retained separately under `video/final/screenshots/`.

```bash
python video/scripts/render_all.py --help
python video/scripts/render_all.py --quality preview
python video/scripts/render_all.py --quality final
python video/scripts/generate_screenshots.py --help
python video/scripts/compose_video.py --help
python video/scripts/validate_layout.py
python -m pytest -q video/tests
```

Render a single final scene with `--only Architecture`, `--only Training`, or another scene name shown by `render_all.py --help`. Composition requires scene renders and writes its MP4 into the ignored render tree. Build output is not needed to use the model or inspect the submission stills.

## Source map

- `scenes/`, `components/`: maintainable Manim source.
- `config/`, `data/`: theme, timings, narration and source metrics.
- `scripts/`: render, compose, validate, inject evidence, and subtitle utilities.
- `vhs/`: VHS source tapes.
- `tests/`: video layout, metric, timing, and screenshot checks.
- `final/screenshots/`: seven selected submission stills.
- `final/narration/`: final narration script and timing manifest.

The frozen duplicate source tree, QA captures, and output bundles under `video/final/` remain local for provenance but are not versioned. Root-level source is the maintained implementation. The repository's main README links the architecture and recovery figures used to summarize the project.
