# LLM-Informed Explainable AI for Software Defect Prediction

This repository contains the reproducibility package for an empirical study of
LLM-informed explainable AI (XAI) for software defect prediction.

The study investigates whether incorporating large language models (LLMs) into
local explanation methods changes explanation quality, how sensitive the
resulting explanations are to decoding parameters, and which prompt components
contribute to the observed behaviour.

The repository structure and research-question numbering are aligned with the
current manuscript.

---

## Study Overview

Four traditional local explanation methods are compared with four directly
matched LLM-informed variants:

| Traditional Method | LLM-Informed Variant |
|---|---|
| LIME | LLM-LIME |
| KernelSHAP | LLM-KernelSHAP |
| LOFO | LLM-LOFO |
| Counterfactual | LLM-Counterfactual |

All matched comparisons use the same frozen predictor, sampled instances, and
predictor feature space so that differences can be attributed as directly as
possible to the LLM-informed explanation mechanism.

---

## Primary LLM

The primary LLM used in the main experiments is:

- **Qwen3.6-27B**
- Hugging Face model ID: `Qwen/Qwen3.6-27B`

Historical experimental outputs may retain the internal identifier

```text
qwen36_27b_local
```

for provenance and backward compatibility.

Before the primary experiments, multiple candidate LLMs were evaluated.
Qwen3.6-27B was selected as the primary locally deployable model because it
provided a comparatively balanced profile across the evaluated explanation
dimensions while supporting controlled and reproducible local execution.

---

## Research Questions

The repository follows the research-question structure used in the current
manuscript.

### RQ1 — Traditional vs. LLM-Informed Explanations

RQ1 evaluates whether LLM-informed explanation methods differ from their
directly matched traditional counterparts.

Explanation quality is evaluated from three complementary dimensions:

#### Stability

The stability dimension evaluates the consistency of explanations across
repeated runs.

Primary metrics include:

- Direction Agreement@5
- Overlap@5
- Rank Agreement@5

#### Model-Response Alignment

The model-response alignment dimension evaluates whether the direction and
magnitude implied by an explanation are consistent with the actual response of
the frozen prediction model when relevant features are perturbed.

Primary metrics include:

- Direction Consistency
- Meaningful-Effect Rate
- Mean `|Δp|`

#### Discriminativeness

The discriminativeness dimension evaluates whether explanations remain
instance-specific rather than collapsing toward highly similar feature sets
across different instances.

Primary metrics include:

- Normalised Feature Entropy
- Pairwise Jaccard
- Instance-IDF Specificity

The RQ1 experiments therefore contain both the traditional explanations and
their matched Qwen3.6-27B-informed variants.

---

### RQ2 — Decoding-Parameter Sensitivity

RQ2 evaluates the sensitivity of LLM-informed explanations to decoding
parameters.

The analysis examines whether changing generation settings produces meaningful
variation in explanation behaviour while keeping the prediction task,
instances, explanation method, and LLM fixed.

The evaluated decoding configurations are defined by the authoritative RQ2
configuration files under:

```text
configs/experiment.rq2_qwen36.yaml
configs/rq2_shards/
```

The corresponding outputs are stored under:

```text
results/rq2/
```

---

### RQ3 — Prompt-Component Ablation

RQ3 evaluates the contribution of individual prompt components through
controlled prompt ablation.

The analysis compares the full prompt against versions in which specific
prompt components are removed while the underlying model, instances, and
explanation procedures remain fixed.

The authoritative RQ3 configuration is:

```text
configs/experiment.rq3_qwen36.yaml
```

Shard-level configurations are stored under:

```text
configs/rq3_shards/
```

The corresponding outputs are stored under:

```text
results/rq3/
```

---

## Repository Structure

```text
llm_xai/
│
├── configs/
│   ├── rq2_shards/
│   ├── rq3_shards/
│   ├── experiment.qwen36_full.yaml
│   ├── experiment.rq1_qwen36_full.yaml
│   ├── experiment.rq2_qwen36.yaml
│   └── experiment.rq3_qwen36.yaml
│
├── data/
│   └── Study datasets and processed data
│
├── experiments/
│   ├── _common.py
│   ├── apply_rq1_qwen36_corrections.py
│   ├── build_qwen36_publication_tables.py
│   ├── build_qwen36_rq1_primary_table.py
│   ├── build_rq2_primary_table.py
│   ├── build_rq3_component_summary.py
│   ├── build_rq3_primary_table.py
│   ├── finalize_rq2_for_rq3.py
│   ├── finalize_rq3_statistics.py
│   ├── finalize_rq5_statistics.py
│   ├── plot_qwen36_rq1_heatmap.py
│   ├── rq23_helpers.py
│   ├── run_comparison.py
│   ├── run_discriminativeness.py
│   ├── run_model_response_alignment.py
│   ├── run_rq2.py
│   ├── run_rq3.py
│   └── run_stability.py
│
├── results/
│   ├── publication_master/
│   ├── rq1/
│   ├── rq2/
│   └── rq3/
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── explainers.py
│   ├── llm.py
│   ├── metrics.py
│   ├── perturbation.py
│   ├── predictor.py
│   ├── threshold_predictor.py
│   └── utils.py
│
├── .gitattributes
├── .gitignore
├── README.md
└── README_REPRODUCIBILITY.md
```

---

## Authoritative Experiment Configurations

The experiment configuration files are stored under `configs/`.

### Primary Qwen3.6-27B configuration

```text
configs/experiment.qwen36_full.yaml
```

### RQ1

```text
configs/experiment.rq1_qwen36_full.yaml
```

### RQ2

```text
configs/experiment.rq2_qwen36.yaml
configs/rq2_shards/
```

### RQ3

```text
configs/experiment.rq3_qwen36.yaml
configs/rq3_shards/
```

Where supported by the experiment loader, an explicit configuration can be
selected through the environment variable:

```bash
export LLM_XAI_EXPERIMENT_CONFIG=<config-file>
```

On Windows PowerShell:

```powershell
$env:LLM_XAI_EXPERIMENT_CONFIG="<config-file>"
```

---

## Data

The study uses software defect prediction data derived from the OpenStack and
Qt datasets.

Study data are stored under:

```text
data/
```

The same frozen dataset splits are used across matched experimental conditions
to avoid introducing sampling differences into comparisons between traditional
and LLM-informed explanations.

Reproduction instructions are maintained at the repository root in
`README_REPRODUCIBILITY.md`; there is no separate `docs/` directory in the
current repository layout.

---

## Core Implementation

The main implementation is located under:

```text
src/
```

Important modules include:

```text
src/data.py
```

Dataset loading and preprocessing.

```text
src/predictor.py
src/threshold_predictor.py
```

Prediction-model interfaces and threshold-based prediction behaviour.

```text
src/explainers.py
```

Traditional and LLM-informed explanation implementations.

```text
src/llm.py
```

LLM interaction and generation logic.

```text
src/perturbation.py
```

Feature-perturbation procedures used for model-response evaluation.

```text
src/metrics.py
```

Evaluation metrics for stability, model-response alignment, and
discriminativeness.

```text
src/utils.py
```

Shared utilities.

---

## RQ1 Experiment Scripts

RQ1 contains the primary matched comparison between traditional and
LLM-informed explanations.

The three evaluation dimensions can be executed and analysed using:

```text
experiments/run_stability.py
experiments/run_model_response_alignment.py
experiments/run_discriminativeness.py
```

Matched traditional-versus-LLM comparison procedures are implemented in:

```text
experiments/run_comparison.py
```

Additional RQ1 processing and publication-table generation are provided by:

```text
experiments/apply_rq1_qwen36_corrections.py
experiments/build_qwen36_rq1_primary_table.py
experiments/build_qwen36_publication_tables.py
experiments/plot_qwen36_rq1_heatmap.py
```

RQ1 outputs are stored under:

```text
results/rq1/
```

---

## RQ2 Experiment Scripts

RQ2 investigates decoding-parameter sensitivity.

The main execution script is:

```text
experiments/run_rq2.py
```

Primary table generation is performed by:

```text
experiments/build_rq2_primary_table.py
```

RQ2 configurations are stored under:

```text
configs/experiment.rq2_qwen36.yaml
configs/rq2_shards/
```

RQ2 outputs are stored under:

```text
results/rq2/
```

---

## RQ3 Experiment Scripts

RQ3 investigates prompt-component ablation.

The main execution script is:

```text
experiments/run_rq3.py
```

Post-processing and statistical analysis are supported by:

```text
experiments/build_rq3_component_summary.py
experiments/build_rq3_primary_table.py
experiments/finalize_rq3_statistics.py
```

Shared RQ2/RQ3 utilities are located in:

```text
experiments/rq23_helpers.py
```

RQ3 configurations are stored under:

```text
configs/experiment.rq3_qwen36.yaml
configs/rq3_shards/
```

RQ3 outputs are stored under:

```text
results/rq3/
```

---

## Publication Outputs

Publication-oriented summaries are stored under:

```text
results/publication_master/
```

These files provide the bridge between the experimental outputs and the
quantitative values reported in the manuscript.

Where possible, manuscript tables should be generated from these derived
outputs rather than transcribed manually.

---

## Reproducibility

Detailed reproduction instructions are provided in:

```text
README_REPRODUCIBILITY.md
```

This document should be consulted for:

- environment setup;
- model requirements;
- dataset preparation;
- experiment execution;
- configuration selection;
- output generation; and
- reproduction of publication-level results.

---

## Experimental Controls

The main experiments are designed to isolate the effect of the LLM-informed
explanation mechanism.

Matched traditional and LLM-informed conditions use the same:

- frozen prediction models;
- datasets and data splits;
- sampled instances;
- predictor feature space;
- explanation-method pairing; and
- evaluation procedures.

For experiments involving stochastic LLM generation, repeated runs and fixed
experimental configurations are used to support controlled comparison.

---

## Reproducibility Scope

The repository focuses on the experiments and outputs used in the current
manuscript.

The primary reproducibility package excludes experimental artefacts that are
not part of the final reported analysis, such as:

- smoke tests;
- connectivity tests;
- failed infrastructure runs;
- incomplete runs;
- superseded experimental protocols; and
- exploratory outputs not used in the manuscript.

This separation is intended to prevent non-primary experimental outputs from
being confused with the results reported in the paper.

---

## Citation

If you use this repository, please cite the corresponding paper.

Citation information will be updated when the manuscript receives its final
publication metadata.

---

## License

Please refer to the repository license information for permitted use and
redistribution.