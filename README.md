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

The primary LLM used in the study is:

- **Qwen3.6-27B**
- Hugging Face model ID: `Qwen/Qwen3.6-27B`

Historical experimental outputs retain the internal identifier
`qwen36_27b_local` for provenance.

## Research Questions

- **RQ1:** Explanation stability
- **RQ2:** Model-response alignment
- **RQ3:** Explanation discriminativeness
- **RQ4:** Traditional vs. matched LLM-informed explanations
- **RQ5:** Decoding-parameter sensitivity
- **RQ6:** Prompt-component ablation

## Repository Structure

```text
src/                  Core implementation
experiments/          Experiment and analysis scripts
configs/              Frozen experiment configurations
data/                 Raw and processed OpenStack / Qt data
results/rq1_rq3/      RQ1-RQ3 archived outputs
results/rq4/          RQ4 paired-comparison outputs
results/rq5/          RQ5 final aggregated outputs
results/rq6/          RQ6 final aggregated outputs
results/publication_master/
docs/                 Checksums and environment information
```

## Environment

The final experiments used **Python 3.13.0**.

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

A less restrictive dependency specification is also available in
`requirements.txt`.

## Authoritative Configurations

Use the following frozen configurations for reproduction.

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

The experiment loader supports explicit configuration selection through:

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

Dataset integrity can be checked with:

```bash
sha256sum -c docs/SHA256_DATA.txt
```

## Frozen Predictors

Full reproduction requires:

```text
models/openstack_predictor.pkl
models/qt_predictor.pkl
```

These files are not committed as ordinary Git objects because they exceed
GitHub's standard per-file size limit. They are included in the full
reproducibility artifact described below.

Expected predictor checksums are recorded in:

```text
docs/PREDICTOR_CHECKSUMS.txt
```

## Full Reproducibility Artifact

The complete reproducibility package is distributed through the GitHub Release:

**Full Reproducibility Artifact v1.0**  
Tag: `v1.0-reproducibility`

Release page:

https://github.com/drleipei/llm_xai/releases/tag/v1.0-reproducibility

The release contains:

- frozen OpenStack and Qt predictors
- complete RQ1-RQ6 reproducibility materials
- full RQ5 32-shard outputs
- full RQ6 32-shard outputs
- authoritative configurations
- source code
- processed/raw study data
- integrity checksums

Main archive:

```text
llm_xai_full_reproducibility_artifact_20260924.tar.gz
```

SHA256:

```text
d9be6d9a6f045f8b21762a91483bbb88bcbe4addfeff24aff35ace7f1b587a8a
```

The accompanying checksum file is:

```text
llm_xai_full_reproducibility_artifact_20260924.tar.gz.sha256
```

Verify on Linux/macOS:

```bash
sha256sum -c llm_xai_full_reproducibility_artifact_20260924.tar.gz.sha256
```

Verify on Windows PowerShell:

```powershell
Get-FileHash .\llm_xai_full_reproducibility_artifact_20260924.tar.gz -Algorithm SHA256
```

The resulting hash should match the SHA256 value above.

> **Note:** If this repository is private, the release is accessible only to
> users who have permission to access the repository.

## RQ1-RQ3 Integrity Verification

Archived RQ1-RQ3 outputs can be verified with:

```bash
sha256sum -c docs/SHA256_RQ1_RQ3.txt
```

## RQ4

RQ4 compares each traditional method with its matched LLM-informed variant.

Final outputs are available under:

```text
results/rq4/
```

## RQ5

RQ5 evaluates six decoding configurations:

```text
T=0.0, top_p=1.0
T=0.2, top_p=1.0
T=0.5, top_p=1.0
T=0.8, top_p=1.0
T=0.2, top_p=0.9
T=0.2, top_p=0.8
```

The Git repository contains the final aggregated RQ5 outputs under:

```text
results/rq5/
```

The complete 32-shard raw outputs are included in the full reproducibility
artifact.

## RQ6

RQ6 evaluates five prompt conditions:

```text
full
no_semantics
no_grounding
no_constraints
no_instance_context
```

The decoding setting is fixed at:

```text
temperature = 0.2
top_p = 1.0
```

The Git repository contains the final aggregated RQ6 outputs under:

```text
results/rq6/
```

The complete 32-shard raw outputs are included in the full reproducibility
artifact.

## Publication Tables

Publication-oriented summaries are available under:

```text
results/publication_master/
```

These files provide the bridge between archived experimental outputs and the
values reported in the manuscript.

## Reproducibility Scope

This repository focuses on the primary Qwen3.6-27B experiment.

The following are intentionally excluded from the primary reproducibility
package:

- smoke tests
- connectivity tests
- failed infrastructure runs
- partial runs
- superseded protocol runs
- invalid BAD_80INSTANCE RQ6 shards
- auxiliary multi-LLM matched experiments

These exclusions prevent exploratory or non-primary outputs from being
confused with the results reported in the manuscript.
