# Git isolation for post-hackathon research

On 2026-10-02, the active research workspace at
`/home/insightlabs38/latticelm` was detached from Git without changing its
files or the running systemd services. A normal `git` command in that
directory no longer finds a repository. This makes accidental GitHub commits
or pushes from the research workspace impossible.

The frozen hackathon source has a separate clean checkout at
`/home/insightlabs38/latticelm-frozen`, at commit
`43832b4c7bfe8be007ca6b9d1042aa2041c7f280`. Its `origin` fetch URL
remains the historical GitHub repository, while its push URL is a nonexistent
local path. The original Git metadata was preserved at
`/home/insightlabs38/latticelm-preisolation-git-metadata` for forensic
recovery. Neither copy is used by the research controller.

The post-hackathon code, registry, results, and checkpoints remain at their
original absolute paths, so in-progress checkpoint/resume and service working
directories are unaffected. The immutable source commit is read from
`FROZEN_HACKATHON_STATE.json`; the controller refuses to create a new deadline
if its state file disappears while registry or event history exists.

Research artifacts are uploaded only to the private Hugging Face model
repository `insightlabs38-pixel/LatticeLM-research` under
`post-hackathon/2026-10-01/`. Authentication comes from the user's local
Hugging Face credential cache, which is mode 0600. No credential is stored in
the research source, service units, reports, or upload receipts.
