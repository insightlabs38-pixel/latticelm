# Future 100M–1B-token data scaling plan

This plan prepares a separate corpus generation path. It does not change DATA-C and no bulk corpus download was started in Phase 7D.

## Recommended source pool

Use a source-balanced pool rather than one undifferentiated web stream. A first 1B-token target can reserve roughly 45–55% for new FineWeb-Edu documents, 10–20% for separately processed English Wikipedia, 10–15% for educational and scientific material, 10–15% for narrative/general prose, and the remainder for high-quality general and procedural web text. These are starting mixture bounds for a future predeclared experiment, not tuned weights.

- **FineWeb-Edu, new non-overlapping shards:** primary educational/general source. Its official card reports a 1.3T-token score-3 pool, ODC-By 1.0 licensing, Common Crawl terms, and recommends separately processed Wikipedia for cleaner formatting ([dataset card, pinned v1.0.0](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/v1.0.0/README.md)). Pin a revision and exclude every existing DATA-C document ID/hash.
- **FineWeb:** optional general and procedural complement, sampled by explicit crawl, language/quality bounds, and deterministic document hash. The official release describes cleaned and deduplicated English Common Crawl data under ODC-By 1.0 ([FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb)). Reuse the published [DataTrove FineWeb pipeline](https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py) where practical and record every deviation.
- **English Wikipedia:** encyclopedic prose from a dated `pages-articles` dump, namespace 0 only, excluding talk/user/template pages and benchmark matches. Wikimedia states that original dump text is generally available under CC BY-SA 4.0 and GFDL, subject to project-specific terms and possible undetected infringements ([Wikimedia dump licensing](https://dumps.wikimedia.org/legal.html)). Preserve page ID, revision ID, dump date, title, and attribution data.
- **Open educational resources:** prefer per-book sources with machine-readable text and explicit licenses. OpenStax is useful for science, mathematics, and physical-world explanations, but licenses vary by title; current help material describes CC BY-NC-SA terms for textbooks, while some individual books state CC BY. Record and enforce the license on each edition rather than assuming one collection-wide license ([OpenStax licensing help](https://help.openstax.org/s/article/Licensing-information-of-OpenStax-textbooks)).
- **Project Gutenberg:** useful for narrative and broad language only after per-item copyright checks and removal of boilerplate. Project Gutenberg says most works are unrestricted under US copyright law but some are permission-based copyrighted works, and non-US use needs separate review ([permissions policy](https://www.gutenberg.org/policy/permission), [license](https://www.gutenberg.org/policy/license)). Retain catalog IDs and rights metadata; exclude any item whose status is ambiguous for the intended use.
- **Dolma components:** investigation-only fallback for academic or encyclopedic diversity. The official card exposes source-level provenance and a reproducible toolkit, but version/license presentation has changed and source content retains separate rights considerations ([Dolma card](https://huggingface.co/datasets/allenai/dolma)). Admit only a pinned component after a documented license and overlap review; do not ingest the aggregate blindly.

Exclude synthetic teacher outputs and all known HellaSwag, ARC-Easy, PIQA, WinoGrande, WikiText test, and project common-validation examples. Do not use benchmark training splits merely to raise evaluation scores until competition rules are reviewed explicitly.

## Offline shard format

Store loss-bearing token IDs in immutable little-endian `int32` files of 8–32M tokens each (32–128 MiB). One billion tokens occupies about 4.0 GB before metadata and raw-source retention. Pair each token shard with:

- a JSON manifest containing schema version, source, upstream URI/revision, acquisition timestamp, transformation commit/config hashes, tokenizer hash, token count, document count, byte count, and SHA-256;
- document-boundary offsets and stable document IDs;
- compact source/license labels aligned to document boundaries;
- quality, deduplication-cluster, and decontamination decision records;
- a shard-level validation split assignment derived from stable document hashes.

Training opens read-only NumPy memmaps and requires no network. A top-level manifest lists shards in canonical order and includes the SHA-256 of every child manifest. Validate all hashes before the first step and whenever a run resumes.

## Determinism and exact resume

Use the existing affine without-replacement block permutation independently per source shard, extended with a counter-based source selector. A frozen mixture schedule maps each global batch index to exact per-source sequence counts. Serialize global loss-bearing tokens, optimizer step, per-source draw counts, shard/epoch positions, source-selector counter, Python/PyTorch RNG states, and the top-level manifest hash. The recovery check must reconstruct and hash the next full batch before optimizer state is accepted.

For mixtures that cannot be represented exactly in one eight-sequence batch, predeclare a short deterministic cycle of batch compositions whose aggregate ratio is exact. Keep effective optimization batch fixed across compared models.

## Deduplication and decontamination

Apply normalization and exact SHA-256 document deduplication first, then paragraph-level exact deduplication, then MinHash/LSH near-duplicate clustering. Run this globally across old DATA-C and every future source so Wikipedia/Gutenberg material already present in BabyLM is not counted as unique. Keep one deterministic representative using a frozen source-priority rule.

Build a versioned contamination registry from official evaluation inputs where permitted. Match normalized exact strings, long token n-grams, and near duplicates before tokenization; quarantine the entire matching document and record rule, benchmark, match span/hash, and pipeline version. Keep protected evaluation material outside training storage and test the decontaminator with synthetic fixtures rather than copying benchmark examples into tests.

## Staged implementation

1. Define schema v1, source/license vocabulary, canonical normalization, and top-level manifest validation.
2. Produce a 1–5M-token multi-source fixture and prove shard hashes, mmap reads, deterministic sampling, and cross-process next-batch identity.
3. Run global exact/near deduplication and benchmark decontamination on a 25–50M unique-token pilot; audit a stratified document sample per source.
4. Materialize 100M unique tokens locally and exercise interruption/resume without internet access.
5. Expand toward 1B only after disk sizing, license review, source-mixture freeze, and pilot quality metrics pass. Preserve raw acquisition manifests even if raw text is moved to cheaper storage.

Success means a clean machine can verify the manifest, mmap the shards, reproduce an arbitrary batch by global batch index, and resume with the same next-batch hash without contacting the internet.
