# Artifact Manifest

## Main Repository

The main repository contains:

- `src/` — core implementation
- `experiments/` — RQ execution and analysis scripts
- `configs/` — authoritative experiment configurations and 32 RQ5 / 32 RQ6 shard configs
- `data/` — study data used by the packaged experiments
- `results/rq1_rq3/` — archived RQ1-RQ3 explanations, evaluations, and metrics
- `results/rq4/` — final RQ4 paired-comparison outputs
- `results/rq5/` — final aggregated RQ5 outputs
- `results/rq6/` — final aggregated RQ6 outputs
- `results/publication_master/` — publication-oriented tables and summaries
- `docs/` — checksums and environment notes
- `requirements.txt`
- `requirements-frozen.txt`

## Separate Full Reproducibility Artifact

Release tag:

```text
v1.0-reproducibility
```

Archive:

```text
llm_xai_full_reproducibility_artifact_20260924.tar.gz
```

SHA256:

```text
d9be6d9a6f045f8b21762a91483bbb88bcbe4addfeff24aff35ace7f1b587a8a
```

The full archive additionally contains:

- `models/openstack_predictor.pkl`
- `models/qt_predictor.pkl`
- `results/rq5_shards/`
- `results/rq6_shards/`

Predictor checksums are documented in:

```text
docs/PREDICTOR_CHECKSUMS.txt
```

## Explicitly Excluded From the Primary Artifact

The following classes of outputs are intentionally excluded:

- smoke tests
- connectivity tests
- failed infrastructure runs
- partial runs
- superseded protocol runs
- invalid `BAD_80INSTANCE` RQ6 shards
- auxiliary multi-LLM matched experiments

These exclusions prevent non-primary or invalid runs from being confused with
the results reported in the manuscript.

## Integrity

The repository includes:

- `docs/SHA256_DATA.txt`
- `docs/SHA256_RQ1_RQ3.txt`
- `docs/PREDICTOR_CHECKSUMS.txt`

The full release additionally provides an archive-level SHA256 file.
