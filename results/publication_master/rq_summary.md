# Research Question Summary

| RQ | Primary Question | Experimental Focus | Main Analysis | Main Takeaway |
|---|---|---|---|---|
| RQ1 | How stable are the explanations? | Repeated-run explanation stability | Stability metrics across methods, datasets, and prediction categories | Evaluates reproducibility of traditional and LLM-informed explanations |
| RQ2 | Do the explanations align with frozen-model behavior? | Independent perturbation response | Direction consistency, meaningful effects, probability changes | Tests whether selected features correspond to measurable predictor responses |
| RQ3 | Are explanations instance-specific and discriminative? | Within- vs between-instance explanation structure | Entropy, Jaccard, IDF specificity, separability | Distinguishes stable explanations from generic explanations |
| RQ4 | Are observed method differences statistically supported? | Matched comparisons | Paired Wilcoxon tests with multiplicity correction | Provides inferential evidence for method-level differences |
| RQ5 | How sensitive are LLM-informed explanations to decoding parameters? | Temperature and top-p sensitivity | Friedman, Kendall's W, planned Wilcoxon-Holm comparisons | Temperature substantially affects stability/diversity; top-p 0.8–1.0 has limited influence |
| RQ6 | Which prompt components influence explanation quality? | One-component-at-a-time prompt ablation | Friedman, Kendall's W, FULL-vs-ablation Wilcoxon-Holm | Instance context shows the strongest observed ablation pattern; grounding has comparatively limited influence |
