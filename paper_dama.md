# Distributed Associative Memory Emerges from the Uncertainty Subspace in Transformer LLMs

**Rodrigo Campos Vargas**
NE-OS AI Research
July 2026

---

## Abstract

Prior work established that hallucination in large language models (LLMs) is geometrically detectable: a compact 100-dimensional subspace within layer 12 of Gemma-2B achieves AUC 0.885 on a binary correct/hallucinated classification, yielding a 41× compression over the full 3072-dimensional hidden state. We hypothesize that this subspace is not monolithic but _distributed_: it decomposes into M=16 specialized agents of 64 dimensions each, where each agent encodes uncertainty within a distinct semantic domain (geography, chemistry, history, etc.). We call this structure **DAMA** (Distributed Associative Memory Architecture). Formally, DAMA is an ensemble of lightweight probes whose outputs are recombined by a learned decoder, analogous to a content-addressable associative memory where fragments from distributed agents converge into a unified recall signal. We derive a falsifiable hypothesis: 16 probes trained on non-overlapping 64-dimensional subspaces will reconstruct AUC ≥ 0.885, matching the monolithic baseline. If confirmed, DAMA offers a mechanistic account of domain-specific confidence calibration and a direct path to continual learning without catastrophic forgetting—updating a single agent leaves all others intact.

---

## 1. Introduction

Transformer-based LLMs encode rich internal representations, but the spatial organization of uncertainty within those representations remains poorly understood. Recent geometric analyses of hallucination (see Section 2) demonstrate that model confidence is not diffusely encoded across all dimensions: instead, it concentrates in a low-dimensional subspace that is identifiable via variance-based feature selection and logistic probing.

A natural follow-up question arises: _is this subspace itself monolithic, or does it have internal structure?_ The present work proposes that it is structured as a set of M specialized sub-modules, each attending to a distinct semantic domain. This framing connects to classical theories of distributed associative memory (Hopfield, 1982; Kanerva, 1988) and to modern mixture-of-experts architectures, but operates at the level of the internal geometry of a single layer's hidden state—no architectural modification is required.

The key contributions of this paper are:
1. A formal definition of DAMA and its relationship to prior uncertainty geometry findings.
2. A falsifiable experimental design executable on commodity hardware using pre-extracted hidden states.
3. A theoretical account of why distributed specialization enables continual learning without interference.

---

## 2. Background

### 2.1 Uncertainty Geometry (Hallucination Subspace)

In the companion paper _"Hallucination as Geometric Collapse in the Residual Stream"_ (NE-OS Paper 1), we showed that the final-token hidden state at layer 12 of Gemma-2B encodes a robust uncertainty signal. A logistic probe trained on the top-100 principal components of the hidden state achieves AUC 0.885 on binary hallucination detection. This represents a 41× compression: 100 of 3072 dimensions suffice to capture the signal.

### 2.2 Three Phases of Residual Stream Processing (Paper 2)

_"Three Phases of Internal Processing in Transformer LLMs"_ (NE-OS Paper 2) identified three computational regimes across layers:
- **Region A** (L1–L8): token-level feature construction
- **Region B** (L9–L23): semantic integration and associative recall
- **Region C** (L24–L28): output decoding and vocabulary projection

DAMA operates in Region B. The uncertainty subspace at L12 sits at the beginning of this region, consistent with the hypothesis that associative recall of domain-specific knowledge is what drives divergence between correct and hallucinated outputs.

### 2.3 Engram and RCLA (Paper 3)

_"Residual Crystallization as Latent Anchoring"_ (NE-OS Paper 3) introduced the Engram: a monolithic latent vector that anchors factual recall across layers via residual stream accumulation. DAMA is the natural successor: where the Engram is a single fixed point, DAMA distributes memory across M specialized engrams, each sovereign over a semantic domain.

### 2.4 Residual Stream Accumulation (Paper 2 Mechanism)

The residual stream accumulates information additively across layers. DAMA's recall signal—a weighted sum of M agent outputs—is structurally identical to one step of residual accumulation, suggesting the architecture is a learned approximation of the transformer's own internal dynamics.

---

## 3. The DAMA Architecture

### 3.1 Formal Definition

Let $\mathbf{h} \in \mathbb{R}^D$ be the hidden state at the final token position, layer $L$, where $D = 3072$ for Gemma-2B. Let $\mathbf{v} \in \mathbb{R}^{d_0}$ ($d_0 = 100$) be the uncertainty subspace projection (top-$d_0$ PCA components).

DAMA partitions this subspace into M agents:

$$\mathbf{f}_i = W_i \mathbf{h} \in \mathbb{R}^{d}, \quad i = 1, \ldots, M$$

where $W_i \in \mathbb{R}^{d \times D}$ is agent $i$'s projection matrix, $d = 64$, and $M = 16$. Each agent produces a scalar confidence estimate:

$$p_i = \sigma(\mathbf{w}_i^\top \mathbf{f}_i + b_i)$$

The decoder assigns relevance weights based on a query derived from the uncertainty signal:

$$w_i = \text{softmax}_i\left(\mathbf{q}^\top \mathbf{f}_i / \sqrt{d}\right)$$

where $\mathbf{q} \in \mathbb{R}^d$ is a learned query vector. The final recall is the weighted combination:

$$\hat{p} = \sum_{i=1}^{M} w_i \cdot p_i$$

### 3.2 Specialization Hypothesis

We hypothesize that training causes agents to specialize: agent $i$ develops high discriminative power on a subset of semantic domains and near-chance performance on others. Specialization is measured by the per-domain AUC profile:

$$\text{spec}(i) = \max_c \text{AUC}(p_i, \mathbf{y}_c) - \text{mean}_c \text{AUC}(p_i, \mathbf{y}_c)$$

where $\mathbf{y}_c$ are binary labels for category $c$.

### 3.3 Relationship to Associative Memory

In a Hopfield network, memories are stored as attractors in a high-dimensional space and retrieved via energy minimization. DAMA implements a soft analogue: each agent stores a distributed fragment of a semantic domain's uncertainty pattern, and the decoder's attention mechanism performs content-addressable retrieval—given a query (the input's uncertainty signal), the decoder identifies which agents' fragments are relevant and combines them into a recall.

The circle metaphor is precise: M agents stand on the periphery of the uncertainty subspace, each holding a fragment. The query stands at the center and calls; each agent responds with its fragment weighted by its relevance. The recall emerges from the superposition.

---

## 4. Experimental Design

### 4.1 Data

We use pre-extracted hidden states from Gemma-2B layer 12, available as `glw_hs.npz` (N=950 samples, D=3072). Binary labels (correct=0, hallucinated=1) and semantic category annotations are available from `gemma_layerwise_ckpt.json`.

### 4.2 Baseline: Monolithic Probe

1. Extract $\mathbf{h}^{(12)}$ for all N samples.
2. Compute PCA on training split; retain top-100 components.
3. Train logistic regression on PCA-projected vectors.
4. Report AUC on held-out test split (80/20 stratified split).

This replicates the AUC 0.885 result from the hallucination geometry paper.

### 4.3 DAMA Experiment

**Subspace decomposition (two variants):**

*Variant A — Dimension partition:* Divide the 4096 (or 3072) raw dimensions into M=16 non-overlapping groups of equal size. Each agent trains a logistic probe on its group's dimensions projected to 64 dimensions via local PCA.

*Variant B — PCA subspace partition:* Compute top-100 global PCA components; partition these 100 components into 16 non-overlapping groups of ~6 components each. Each agent trains a logistic probe on its 6-component slice.

**Decoder:**

The ensemble decoder weights each agent by its individual AUC on the training split (calibrated weighting):

$$w_i = \frac{\text{AUC}_i^{\text{train}} - 0.5}{\sum_j (\text{AUC}_j^{\text{train}} - 0.5)}$$

Agents with AUC ≤ 0.5 are assigned weight 0. The final ensemble prediction is $\hat{p} = \sum_i w_i p_i$.

**Falsifiable hypothesis:**

> DAMA ensemble AUC (Variant A or B) ≥ 0.885 on the held-out test set.

### 4.4 Specialization Analysis

For each agent $i$ and each semantic category $c \in \{\text{technology, science, geography, history, ...}\}$:
- Compute $\text{AUC}(p_i, \mathbf{y}_c)$ on the subset of samples belonging to category $c$.
- Identify each agent's dominant domain: $c^* = \arg\max_c \text{AUC}(p_i, \mathbf{y}_c)$.
- Visualize as a heatmap (agents × domains).

If agents specialize, the heatmap will be approximately block-diagonal with one high-AUC cell per row.

### 4.5 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| AUC (macro) | Overall ensemble vs. monolithic probe |
| Specialization score | Mean of per-agent domain AUC variance |
| Hypothesis verdict | CONFIRMED if DAMA AUC ≥ 0.885 |

---

## 5. Expected Results

We expect three outcomes:

**E1 (Hypothesis confirmed):** DAMA ensemble AUC ≥ 0.885. This would demonstrate that the 100-dimensional uncertainty subspace is internally decomposable without loss of discriminative power.

**E2 (Specialization observed):** At least 8 of 16 agents show a dominant domain with $\Delta\text{AUC} > 0.05$ over their mean across domains.

**E3 (Residual stream consistency):** The weighted recall $\hat{p}$ correlates more strongly with Layer 12 activations than with Layer 8 or Layer 20, consistent with Region B being the locus of associative recall.

If E1 fails but E2 holds, the result is still meaningful: specialization exists, but the 16-agent decomposition at 64 dims is not sufficient—suggesting either M or d requires adjustment.

---

## 6. Implications for Continual Learning

The most consequential implication of DAMA concerns continual learning. Current LLMs exhibit catastrophic forgetting: updating weights to encode new knowledge overwrites old knowledge (McCloskey & Cohen, 1989; Kirkpatrick et al., 2017).

If uncertainty about domain $c$ is primarily encoded by agent $i_c$, and agents operate on disjoint subspaces of the hidden state, then:
- Fine-tuning on new data in domain $c'$ updates only $W_{i_{c'}}$, $\mathbf{w}_{i_{c'}}$, $b_{i_{c'}}$.
- All other agents remain unchanged.
- Prior knowledge in domains $c \neq c'$ is preserved.

This is not merely a routing trick (cf. mixture-of-experts): the specialization emerges from the geometry of the pre-trained model's own uncertainty representation. DAMA is a _readout_ of a structure that already exists; it does not require retraining the base model.

---

## 7. Limitations

**L1 — Gemma-2B specificity.** All experiments use Gemma-2B layer 12. Generalization to Qwen2.5-7B, Mistral-7B, and other architectures requires replication experiments with comparable hidden state extractions.

**L2 — Category label quality.** The semantic category labels (`cats` field) are coarse and may not align with the domains agents naturally specialize in. A mismatch would suppress E2 even if specialization exists at a finer granularity.

**L3 — Subspace partition heuristic.** Dividing dimensions by index (Variant A) is arbitrary; agents may span natural subspaces that cross index boundaries. Learned partitioning (e.g., via sparse dictionary learning) would be a stronger test.

**L4 — AUC-weighted decoder.** The decoder is not learned end-to-end; it uses training AUC as a proxy for relevance. A learned attention decoder might recover higher AUC from the same agents.

**L5 — N=950.** The sample size is modest; confidence intervals on per-agent per-domain AUC will be wide for rare categories.

---

## 8. Conclusions

We have introduced DAMA, a Distributed Associative Memory Architecture that hypothesizes the uncertainty subspace of transformer LLMs to be internally structured as M=16 specialized agents, each sovereign over a distinct semantic domain. The falsifiable core claim—that 16 probes on 64-dimensional subspaces match or exceed the monolithic 100-dimensional probe at AUC 0.885—can be tested on commodity hardware using existing hidden state extractions.

If confirmed, DAMA provides: (a) a mechanistic account of domain-specific calibration in LLMs, (b) a direct path to modular continual learning without catastrophic forgetting, and (c) a new tool for interpretability—reading out which agent activates reveals what semantic domain the model "thinks" the question belongs to.

The circle of agents awaits. The experiment will determine whether they each carry a distinct fragment, or whether the memory is truly monolithic.

---

## References

- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*, 79(8), 2554–2558.
- Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521–3526.
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks. *Psychology of Learning and Motivation*, 24, 109–165.
- NE-OS Paper 1: "Hallucination as Geometric Collapse in the Residual Stream" (2026).
- NE-OS Paper 2: "Three Phases of Internal Processing in Transformer LLMs" (2026).
- NE-OS Paper 3: "Residual Crystallization as Latent Anchoring" (2026).
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*, 30.
