# LLM-Informed Explainable AI for Software Defect Prediction

This repository contains the reproducibility package for an empirical study of
LLM-informed explainable AI methods for software defect prediction.

## Methods

Four traditional explanation methods are compared with four matched
LLM-informed variants:

| Traditional | LLM-informed |
|---|---|
| LIME | LLM-LIME |
| KernelSHAP | LLM-KernelSHAP |
| LOFO | LLM-LOFO |
| Counterfactual | LLM-Counterfactual |

## Primary LLM

The primary LLM is:

- Qwen3.6-27B
- Hugging Face model ID: `Qwen/Qwen3.6-27B`

Historical experimental outputs retain the internal identifier
`qwen36_27b_local` for provenance.

## Research Questions

- RQ1: Explanation stability
- RQ2: Model-response alignment
- RQ3: Explanation discriminativeness
- RQ4: Traditional vs. matched LLM-informed explanations
- RQ5: Decoding-parameter sensitivity
- RQ6: Prompt-component ablation

## Repository Structure

```text
src/                  Core implementation
experiments/          Experiment and analysis scripts
configs/              Frozen experiment configurations
data/                 Raw and processed OpenStack / Qt data
results/rq1_rq3/      RQ1-RQ3 outputs
results/rq4/          RQ4 paired-comparison outputs
results/rq5/          RQ5 final aggregated outputs
results/rq6/          RQ6 final aggregated outputs
results/publication_master/
docs/                 Checksums and environment information
```

## Environment

The final experiments used Python 3.13.0.

Install the exact frozen environment with:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-frozen.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-frozen.txt
```

## Authoritative Configurations

RQ1-RQ3:

`configs/experiment.qwen36_full.yaml`

RQ4:

`configs/experiment.rq4_qwen36_full.yaml`

RQ5:

`configs/experiment.rq5_qwen36.yaml`

RQ6:

`configs/experiment.rq6_qwen36.yaml`

The configuration can be selected using:

```bash
export LLM_XAI_EXPERIMENT_CONFIG=<config-file>
```

## Data

The repository contains the OpenStack and Qt study data under:

```text
data/raw/
data/processed/
```

Primary evaluation files:

```text
data/processed/openstack/test.csv
data/processed/qt/test.csv
```

## Frozen Predictors

Full reproduction requires:

```text
models/openstack_predictor.pkl
models/qt_predictor.pkl
```

These files are distributed separately in the full reproducibility artifact
because they exceed GitHub's standard per-file size limit.

Expected SHA256 values are recorded in:

`docs/PREDICTOR_CHECKSUMS.txt`

## Integrity Verification

Dataset integrity:

```bash
sha256sum -c docs/SHA256_DATA.txt
```

RQ1-RQ3 archived-result integrity:

```bash
sha256sum -c docs/SHA256_RQ1_RQ3.txt
```

## RQ5 and RQ6 Raw Shards

The main GitHub repository contains the final aggregated RQ5 and RQ6 outputs.

The complete 32-shard raw outputs for RQ5 and RQ6 are distributed separately
with the full reproducibility artifact.

## Publication Tables

Publication-oriented summaries are available under:

```text
results/publication_master/
```

These files provide the bridge between archived experimental outputs and the
values reported in the manuscript.

## Scope

This repository contains the primary Qwen3.6-27B study.

Smoke tests, failed infrastructure runs, partial runs, superseded protocol
runs, and auxiliary multi-LLM experiments are intentionally excluded from the
primary reproducibility package.
