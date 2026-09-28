# Reproducibility Guide

## Scope

This package reproduces the primary Qwen3.6-27B study of LLM-informed
explainable AI for software defect prediction.

The current repository and manuscript use three research questions:

- **RQ1:** Traditional vs. matched LLM-informed explanations
- **RQ2:** Decoding-parameter sensitivity
- **RQ3:** Prompt-component ablation

RQ1 evaluates explanation quality across three complementary dimensions:

- **Stability**
- **Model-Response Alignment**
- **Discriminativeness**

The primary LLM is:

- **Qwen3.6-27B**
- Hugging Face model ID: `Qwen/Qwen3.6-27B`
- Historical internal identifier retained in archived outputs:
  `qwen36_27b_local`

Auxiliary multi-LLM experiments, smoke tests, failed infrastructure runs,
partial runs, superseded protocol runs, and exploratory outputs not used in the
manuscript are intentionally excluded from the primary reproducibility scope.

---

## Environment

The final experiments used:

- Linux x86_64
- Python 3.13.0

Create a virtual environment with:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the dependency specification available with the repository or the
full reproducibility artifact.

---

## Repository Layout

The current repository is organised as follows:

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

There is no separate `docs/` directory in the current repository layout.
Reproduction instructions are maintained in this file at the repository root.

---

## Data

Study data are stored under:

```text
data/
```

The study uses frozen OpenStack and Qt software defect prediction data.

The same dataset splits and sampled instances are reused across matched
traditional and LLM-informed conditions to avoid introducing unnecessary
sampling variation into the comparison.

---

## Random Seed and Sampling

The global random seed used by the study is:

```text
20260810
```

Sampling uses 20 instances per prediction category:

```text
TP
TN
FP
FN
```

for each dataset.

This yields:

```text
20 instances × 4 categories = 80 instances per dataset
```

and:

```text
80 instances × 2 datasets = 160 dataset-specific instances
```

for the primary sampling design.

---

## Authoritative Configurations

Use the explicit experiment configurations rather than relying on generic
fallback settings.

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

Where supported by the experiment loader, explicit configuration selection can
be provided through:

```bash
export LLM_XAI_EXPERIMENT_CONFIG=<config-file>
```

On Windows PowerShell:

```powershell
$env:LLM_XAI_EXPERIMENT_CONFIG="<config-file>"
```

---

# RQ1 — Traditional vs. Matched LLM-Informed Explanations

RQ1 compares four traditional explanation methods with four directly matched
LLM-informed variants:

| Traditional | LLM-informed |
|---|---|
| LIME | LLM-LIME |
| KernelSHAP | LLM-KernelSHAP |
| LOFO | LLM-LOFO |
| Counterfactual | LLM-Counterfactual |

Matched conditions use the same:

- frozen prediction model;
- dataset split;
- sampled instance;
- predictor feature space;
- explanation-method pairing; and
- evaluation procedure.

RQ1 is evaluated across three dimensions.

---

## RQ1-A — Stability

Stability evaluates whether repeated explanation runs produce consistent
high-ranked features, directions, and orderings.

Main script:

```text
experiments/run_stability.py
```

Primary metrics include:

- Direction Agreement@5
- Overlap@5
- Rank Agreement@5

Associated RQ1 outputs are stored under:

```text
results/rq1/
```

---

## RQ1-B — Model-Response Alignment

Model-Response Alignment evaluates whether the direction and magnitude implied
by an explanation are consistent with the actual response of the frozen
prediction model under feature perturbation.

Main script:

```text
experiments/run_model_response_alignment.py
```

Primary metrics include:

- Direction Consistency
- Meaningful-Effect Rate
- Mean `|Δp|`

Candidate explanation features are restricted to usable predictor features
before Top-k evaluation. When fewer than the target number of usable unique
features are available, the available features are evaluated without artificial
padding.

Associated RQ1 outputs are stored under:

```text
results/rq1/
```

---

## RQ1-C — Discriminativeness

Discriminativeness evaluates whether explanations remain instance-specific
rather than converging toward highly similar feature sets across instances.

Main script:

```text
experiments/run_discriminativeness.py
```

Primary metrics include:

- Normalised Feature Entropy
- Pairwise Jaccard
- Instance-IDF Specificity

Associated RQ1 outputs are stored under:

```text
results/rq1/
```

---

## RQ1 Matched Comparison

The paired comparison between traditional explanations and their matched
LLM-informed counterparts is implemented in:

```text
experiments/run_comparison.py
```

Additional RQ1 analysis and publication-oriented processing are provided by:

```text
experiments/apply_rq1_qwen36_corrections.py
experiments/build_qwen36_rq1_primary_table.py
experiments/build_qwen36_publication_tables.py
experiments/plot_qwen36_rq1_heatmap.py
```

The comparison outputs are organised under:

```text
results/rq1/
```

---

# RQ2 — Decoding-Parameter Sensitivity

RQ2 evaluates whether LLM-informed explanations are sensitive to decoding
parameters while holding the dataset, instances, explanation method, predictor,
and LLM fixed.

Main script:

```text
experiments/run_rq2.py
```

Primary configuration:

```text
configs/experiment.rq2_qwen36.yaml
```

Shard-level configurations:

```text
configs/rq2_shards/
```

Primary table generation:

```text
experiments/build_rq2_primary_table.py
```

RQ2 outputs are stored under:

```text
results/rq2/
```

The decoding configurations should be taken directly from the authoritative
RQ2 configuration files in the repository.

---

# RQ3 — Prompt-Component Ablation

RQ3 evaluates the contribution of prompt components by comparing the full
prompt with controlled ablation variants.

Main script:

```text
experiments/run_rq3.py
```

Primary configuration:

```text
configs/experiment.rq3_qwen36.yaml
```

Shard-level configurations:

```text
configs/rq3_shards/
```

RQ3 post-processing and analysis are supported by:

```text
experiments/build_rq3_component_summary.py
experiments/build_rq3_primary_table.py
experiments/finalize_rq3_statistics.py
```

Shared RQ2/RQ3 utilities are located in:

```text
experiments/rq23_helpers.py
```

RQ3 outputs are stored under:

```text
results/rq3/
```

The prompt conditions and fixed decoding settings should be taken directly from
the authoritative RQ3 configuration files in the repository.

---

## Publication-Oriented Outputs

Publication tables and summaries are stored under:

```text
results/publication_master/
```

These files provide the bridge between experimental outputs and the values
reported in the manuscript.

Where possible, manuscript tables should be generated from the stored analysis
outputs rather than manually transcribed.

---

## Recommended Reproduction Order

A clean reproduction should follow the dependency structure of the study.

### 1. Prepare the environment

Create and activate the Python environment and install the required
dependencies.

### 2. Prepare data and predictors

Confirm that the OpenStack and Qt data and frozen prediction models required by
the experiment configuration are available.

### 3. Run the RQ1 explanation evaluations

Run:

```text
experiments/run_stability.py
experiments/run_model_response_alignment.py
experiments/run_discriminativeness.py
```

### 4. Run the matched RQ1 comparison

Run:

```text
experiments/run_comparison.py
```

### 5. Generate RQ1 publication outputs

Use the RQ1 analysis and table-generation scripts under `experiments/`.

### 6. Run RQ2

Use:

```text
experiments/run_rq2.py
```

with the RQ2 configuration and shard files.

### 7. Run RQ3

Use:

```text
experiments/run_rq3.py
```

with the RQ3 configuration and shard files.

### 8. Generate publication-level outputs

Use the corresponding build/finalisation scripts and verify the contents of:

```text
results/publication_master/
```

---

## Reproducibility Controls

The experimental design uses several controls intended to isolate the effect of
LLM-informed explanation generation.

Across matched traditional and LLM-informed conditions, the study keeps fixed:

- the prediction model;
- the data split;
- the sampled instances;
- the predictor feature space;
- the matched explanation method; and
- the evaluation procedure.

For stochastic LLM generation, repeated runs and fixed configurations are used
to support controlled comparison.

---

## Reproducibility Scope

The primary reproducibility package focuses on experiments and outputs used in
the current manuscript.

The following are intentionally excluded from the primary reported analysis:

- smoke tests;
- connectivity tests;
- failed infrastructure runs;
- incomplete runs;
- superseded protocol runs; and
- exploratory outputs not used in the manuscript.

This separation prevents non-primary experimental artefacts from being confused
with the results reported in the paper.

---

## Double-Blind Review

Institution-specific hostnames, account paths, and author-identifying metadata
should be omitted from reviewer-facing materials where required by the target
venue.

Repository access, release metadata, and artifact sharing should follow the
double-blind or anonymous-review requirements of the venue to which the
manuscript is submitted.
