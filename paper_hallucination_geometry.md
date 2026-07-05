# Hallucination as Geometric Collapse in the Residual Stream

**Rodrigo Campos Vargas**
NE-OS SpA, Chile — June 2026

**Status:** Submitted to BlackboxNLP 2026 (ACL Workshop)
**OpenReview:** https://openreview.net/forum?id=CPo8Gn7Mi9

---

## Abstract

We show that hallucination in large language models is geometrically detectable before any output token is generated. A linear probe trained on the top-100 principal components of the final-token hidden state at layer 12 of Gemma-2B achieves AUC 0.885 on binary hallucination detection — a 41× compression of the full 4096-dimensional representation with negligible loss. 

The key finding is a **LOCO-CV (Leave-One-Category-Out Cross-Validation)** evaluation: when probes are trained on all categories except one and tested on the held-out category, AUC drops from 0.885 to 0.556. This demonstrates that probes learn **domain identity, not uncertainty** — the signal is not transferable across semantic categories. 

Additional findings: only 100 of 4096 dimensions carry the uncertainty signal (verified via ablation); the signal emerges at the Phase 1/Phase 2 boundary (L12); activation steering at this layer corrects wrong answers without modifying weights.

---

## Key Results

| Metric | Value |
|---|---|
| Probe AUC (full) | 0.885 |
| Probe AUC (LOCO-CV held-out) | 0.556 |
| Compression ratio | 41× (100/4096 dims) |
| Hallucination recall (interceptor) | 75% on unseen prompts |
| Training examples | 435 |
| Activation steering success | Corrects wrong→right without weight modification |

---

## Method

**Dataset:** 435 (prompt, correct/hallucinated) pairs across 13 semantic domains.

**Probe:** Logistic regression on top-100 PCA components of h_{L12} (final token position).

**LOCO-CV:** For each category c, train probe on all other categories, test on c. AUC averaged across categories.

**Interceptor:** Lightweight MLP trained on the 100-dim subspace. Deployed at L12 during inference to flag likely hallucinations before generation completes.

---

## Implications

The LOCO-CV result is a strong negative finding with positive implications: it shows that hallucination detection from internal states is inherently domain-specific. A probe trained on geography questions cannot generalize to chemistry questions. This suggests that the model's internal uncertainty representation is organized by semantic domain — consistent with the DAMA hypothesis (see companion paper).
