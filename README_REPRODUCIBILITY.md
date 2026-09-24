# Reproducibility Guide

## Scope

This package reproduces the primary Qwen3.6-27B study comparing four traditional
explanation methods with four matched LLM-informed variants for software defect
prediction.

Primary LLM:

- Qwen3.6-27B
- Hugging Face ID: `Qwen/Qwen3.6-27B`
- Historical internal identifier retained in archived outputs: `qwen36_27b_local`

Auxiliary multi-LLM experiments, smoke tests, failed infrastructure runs,
partial runs, and superseded protocol runs are intentionally excluded from the
primary artifact.

## Environment

Final experiments used:

- Linux x86_64
- Python 3.13.0

Two dependency specifications are provided:

- `requirements.txt`: project-level compatible dependency ranges
- `requirements-frozen.txt`: exact environment used for the final experiments

Recommended setup:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-frozen.txt
```

## Required Large Artifacts

Full reproduction requires the frozen predictors:

```text
models/openstack_predictor.pkl
models/qt_predictor.pkl
```

Expected SHA256 values are listed in `docs/PREDICTOR_CHECKSUMS.txt`.

The predictors and full RQ5/RQ6 raw shard outputs are included in the separate
full reproducibility archive associated with release tag
`v1.0-reproducibility`.

## Data

Study data are stored under:

```text
data/raw/
data/processed/
```

Primary evaluation files:

```text
data/processed/openstack/test.csv
data/processed/qt/test.csv
```

Verify data integrity with:

```bash
sha256sum -c docs/SHA256_DATA.txt
```

## Random Seed and Sampling

Global random seed:

```text
20260810
```

Sampling uses 20 instances per prediction category (TP, TN, FP, FN) per
dataset, yielding 80 instances per dataset and 160 dataset-specific instances
overall.

## Authoritative Configurations

Use these configurations rather than any generic fallback configuration.

### RQ1-RQ3

```text
configs/experiment.qwen36_full.yaml
```

### RQ4

```text
configs/experiment.rq4_qwen36_full.yaml
```

### RQ5

```text
configs/experiment.rq5_qwen36.yaml
configs/rq5_shards/
```

### RQ6

```text
configs/experiment.rq6_qwen36.yaml
configs/rq6_shards/
```

The configuration loader supports explicit selection with:

```bash
export LLM_XAI_EXPERIMENT_CONFIG=<config-file>
```

## RQ1

Main script:

```text
experiments/run_rq1.py
```

RQ1 evaluates explanation stability using repeated runs per
instance/method. Archived RQ1 outputs are under:

```text
results/rq1_rq3/
```

## RQ2

Main script:

```text
experiments/run_rq2.py
```

RQ2 evaluates up to the five highest-ranked usable unique features returned by
the explanation pipeline. Candidate features are deduplicated and restricted
to the predictor feature space before Top-5 truncation. When fewer than five
usable unique features are available, all available features are evaluated
without artificial padding.

## RQ3

Main script:

```text
experiments/run_rq3.py
```

RQ3 evaluates explanation discriminativeness using entropy, within/between
pairwise Jaccard, instance-IDF specificity, and separability gap.

Verify archived RQ1-RQ3 outputs with:

```bash
sha256sum -c docs/SHA256_RQ1_RQ3.txt
```

## RQ4

Main script:

```text
experiments/run_rq4.py
```

RQ4 operates on completed RQ1-RQ3 metrics and performs paired traditional vs.
matched LLM-informed comparisons. Final outputs are under:

```text
results/rq4/
```

## RQ5

Main script:

```text
experiments/run_rq5.py
```

Six decoding configurations are evaluated:

```text
T=0.0, top_p=1.0
T=0.2, top_p=1.0
T=0.5, top_p=1.0
T=0.8, top_p=1.0
T=0.2, top_p=0.9
T=0.2, top_p=0.8
```

There are 32 dataset-category-method blocks. Final aggregated outputs are under:

```text
results/rq5/
```

The complete raw 32-shard outputs are included in the separate full artifact.

## RQ6

Main script:

```text
experiments/run_rq6.py
```

Five prompt conditions are evaluated:

```text
full
no_semantics
no_grounding
no_constraints
no_instance_context
```

The decoding configuration is fixed at:

```text
temperature = 0.2
top_p = 1.0
```

There are 32 dataset-category-method blocks. Final aggregated outputs are under:

```text
results/rq6/
```

The complete raw 32-shard outputs are included in the separate full artifact.

## Publication-Oriented Outputs

Publication tables and summaries are stored under:

```text
results/publication_master/
```

These files provide the direct bridge between archived experimental outputs
and values reported in the manuscript.

## Double-Blind Review

Institution-specific hostnames, account paths, and author-identifying metadata
are intentionally omitted from the reviewer-facing repository. Repository
access and artifact sharing should follow the anonymity requirements of the
target venue.
