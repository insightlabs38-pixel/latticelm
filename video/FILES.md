# Video files retained in Git

The maintained video implementation lives at the root of `video/`: scene and component source, configuration and source metric JSON, build scripts, VHS tapes, tests, and build instructions. The seven final 1920×1080 submission stills are retained under `final/screenshots/`; final narration text and timing metadata are retained under `final/narration/`.

Generated renders, frame dumps, contact sheets, QA logs and images, preview/composite MP4 files, terminal capture MP4s, local tools, high-resolution duplicate screenshots, ZIP bundles, and the frozen copy of source under `final/` remain on disk but are excluded by `.gitignore`. Rebuild instructions are in [`README.md`](README.md).
