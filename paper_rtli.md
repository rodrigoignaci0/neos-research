# RT-LI: Regional Transformer with Lateral Inhibition — A Three-Phase Architecture for Language Models

**Authors:** Rodrigo Campos Vargas, Claude (Anthropic)
**Affiliation:** NE-OS SpA, Chile
**Date:** July 2026
**Status:** Theoretical proposal — experiments are future work

---

## Abstract

Transformer language models process tokens through a uniform stack of layers, each architecturally identical, despite empirical evidence that fundamentally different computational processes occur at different depths. Recent work (NE-OS Research, 2025a) identifies three distinct phases in transformer forward passes: an early chaos phase (L0–L8), a semantic organization phase (L9–L23), and an abrupt crystallization phase (L24+) where the correct output token emerges with near-certainty. Complementary findings (NE-OS Research, 2025b) show that hallucination-predictive uncertainty concentrates in a 100-dimensional subspace accessible at the phase boundary (L12). These empirical regularities suggest that the linear layer stack is a suboptimal inductive bias: it forces sequential processing of computations that are functionally orthogonal.

We propose **RT-LI (Regional Transformer with Lateral Inhibition)**, an architecture that partitions computation into three parallel specialized regions — Chaos Encoder, Semantic Integrator, and Crystal Decoder — connected by lightweight lateral inhibition gates. Regions operate in parallel rather than sequence, with gate signals propagating bidirectionally between regions to coordinate suppression and activation. We provide a formal specification of the architecture including gate equations, analyze its theoretical properties, and propose a concrete experimental program to validate the design. RT-LI offers potential gains in inference latency (latency proportional to the slowest region rather than the sum of all layers), interpretability (uncertainty subspace localized by design in Region B), and modularity (regions can be scaled independently).

---

## 1. Introduction

The canonical transformer architecture (Vaswani et al., 2017) processes input through a sequential stack of $L$ identical layers. Each layer applies self-attention followed by a feed-forward network with residual connections, and all layers are architecturally interchangeable. This uniformity is computationally convenient but functionally implausible: the computations required to extract surface syntax from raw token embeddings are not the same as those required to organize semantic associations or to commit to a specific output token.

Mechanistic interpretability has revealed that transformer layers are not functionally interchangeable. Early layers encode positional and syntactic features (Tenney et al., 2019); intermediate layers encode semantic content (Geva et al., 2021); late layers perform output selection (nostalgebraist, 2020). Layer-pruning experiments consistently show that removing early layers is disproportionately damaging despite their apparent simplicity (Michel et al., 2019). These findings converge on a picture in which depth is implicitly partitioned into functional regions — but the architecture provides no structural support for this partition.

NE-OS Research (2025a) formalizes this observation by identifying three universal phases in transformer forward passes, validated across Qwen2.5-7B, Mistral-7B, and Gemma-2-2b. Phase 1 (layers 0–8) is a chaos phase: hidden states show high uncertainty, top-1 accuracy is near chance, and representations do not yet cluster by semantic domain. Phase 2 (layers 9–23) is a semantic organization phase: cluster structure emerges, uncertainty decreases (AUC 0.90 for predicting final output), and the residual stream accumulates a metacognitive signal. Phase 3 begins with an abrupt crystallization event: in a mean span of 4.4 layers, top-1 accuracy jumps from ~50% to ~99%. Critically, ablation experiments show that Phase 1 layers are not dispensable — skipping them degrades output quality — demonstrating that all three phases are necessary.

NE-OS Research (2025b) further shows that a 100-dimensional subspace at the Phase 1/Phase 2 boundary (approximately L12) carries sufficient signal to predict hallucinations with AUC 0.885, representing a 41× compression of the full hidden dimension.

These findings motivate a structural question: if LLMs already learn to partition their depth into three functional phases, can we build an architecture that makes this partition explicit? We propose RT-LI, which replaces the linear layer stack with three parallel regions corresponding to the three empirical phases, connected by biologically-inspired lateral inhibition gates. The architecture is designed so that each region specializes in one functional role, the uncertainty subspace is structurally localized, and regions can be executed in parallel during inference.

---

## 2. Background

### 2.1 Three Phases of Transformer Computation

NE-OS Research (2025a) applies the Logit Lens methodology (nostalgebraist, 2020) to map the exact layer at which the correct output token reaches rank 1 in the logit distribution. For each prompt $x$ and layer $l$, the projected logits are:

$$\text{logits}_l = \text{LM\_head}(\text{RMSNorm}(h_l))$$

Real emergence at layer $l^*$ is defined as the first layer where $\text{rank}_l(t^*) < 100$ and this rank does not revert in the immediately following layer. Across 50 prompts in 13 semantic domains, average real emergence occurs at L19.7/28, with domain variation from L13.6 (mathematics) to L26.4 (history).

The three phases are not merely statistical abstractions. Phase 1 representations show no semantic clustering (AUC of uncertainty probe: 0.82). Phase 2 representations exhibit emergent cluster structure with uncertainty AUC rising to 0.90. Phase 3 is characterized by a discontinuous jump — crystallization — that is abrupt rather than gradual. The mean crystallization span of 4.4 layers is inconsistent with a smooth optimization trajectory, suggesting a qualitative change in representational geometry.

### 2.2 Hallucination Geometry and the Uncertainty Subspace

NE-OS Research (2025b) identifies a 100-dimensional subspace of the 4096-dimensional hidden space at L12 that concentrates the signal for hallucination detection. A linear probe trained on this subspace achieves AUC 0.885 on held-out prompts, while the full 4096-dimensional representation achieves AUC 0.891 — a 41× compression with negligible loss. The probe operates at L12, which corresponds to the Phase 1/Phase 2 transition: the earliest layer where the residual stream has accumulated sufficient semantic content for uncertainty to be reliably estimated.

Furthermore, NE-OS Research (2025b) shows that the uncertainty signal in the residual stream is distributed across attention heads redundantly rather than localized to specific heads. This has architectural implications: uncertainty is a property of the full residual stream state, not a circuit-level phenomenon.

### 2.3 Lateral Inhibition in Neuroscience

Lateral inhibition is a canonical mechanism in sensory cortex where activated neurons suppress the activity of adjacent neurons, sharpening response profiles and increasing signal contrast (Hartline et al., 1956). In visual cortex, lateral inhibition produces edge detection; in auditory cortex, it sharpens frequency tuning. The mechanism is implemented by local interneurons that receive excitatory input from principal neurons and project inhibitory connections to neighboring principal neurons.

The functional role of lateral inhibition in cortex is competition resolution: when multiple representations are partially activated, inhibition between them forces a winner-takes-more outcome, producing categorical rather than graded outputs. This is precisely the computation required at the Phase 2 → Phase 3 transition in transformer LLMs, where the model must resolve semantic ambiguity into a specific token.

---

## 3. RT-LI Architecture

### 3.1 Overview

RT-LI replaces the $L$-layer sequential stack with three parallel processing regions:

- **Region A** (Chaos Encoder): $n_A$ transformer layers, processes raw token embeddings
- **Region B** (Semantic Integrator): $n_B$ transformer layers, organizes semantic content
- **Region C** (Crystal Decoder): $n_C$ transformer layers, crystallizes output tokens

The regions receive cross-region signals via lateral inhibition gates: lightweight linear projections that carry suppression or activation signals between regions. Total parameter count is approximately equivalent to a baseline transformer with $n_A + n_B + n_C$ layers, with a small overhead from the gate projections.

Let $D$ be the model hidden dimension, $D_g \ll D$ be the gate dimension, and $T$ be the sequence length. The forward pass proceeds as:

1. Embed input tokens: $h^{(0)} \in \mathbb{R}^{T \times D}$
2. Run all three regions in parallel, with gate signals propagating between them
3. Region C produces the final logits for output token prediction

### 3.2 Region A — Chaos Encoder

Region A consists of $n_A$ standard transformer layers applied to the input embedding:

$$h^A_0 = \text{Embed}(x) + \text{PosEnc}(x)$$
$$h^A_i = \text{TransformerLayer}_i^A(h^A_{i-1}), \quad i = 1, \ldots, n_A$$

The output of Region A is $h^A = h^A_{n_A} \in \mathbb{R}^{T \times D}$.

Region A specializes in low-level feature extraction: tokenization artifacts, positional relationships, surface syntax, and morphological structure. Because it operates on raw embeddings without cross-region input, it can begin processing immediately upon receiving the input. Importantly, based on the ablation results of NE-OS Research (2025a), Region A is not a trivial preprocessing step — its outputs are necessary preconditions for semantic organization in Region B.

### 3.3 Region B — Semantic Integrator

Region B consists of $n_B$ transformer layers that receive both the output of Region A and an inhibitory signal from Region C:

$$h^B_0 = h^A + \text{Gate}_{A \to B}(h^A)$$
$$h^B_i = \text{TransformerLayer}_i^B(h^B_{i-1}), \quad i = 1, \ldots, n_B$$

The $A \to B$ gate is defined as:

$$\text{Gate}_{A \to B}(h^A) = W_{A \to B}^{\text{up}} \cdot \sigma(W_{A \to B}^{\text{down}} \cdot h^A)$$

where $W_{A \to B}^{\text{down}} \in \mathbb{R}^{D_g \times D}$, $W_{A \to B}^{\text{up}} \in \mathbb{R}^{D \times D_g}$, and $\sigma$ is a gating nonlinearity (sigmoid). This gate allows Region A to suppress dimensions of the Region B input that Region A has already resolved with high confidence, preventing redundant recomputation.

Additionally, Region B receives a feedback signal from Region C (the crystallization feedback gate):

$$h^B_{\text{in}} = h^A + \text{Gate}_{A \to B}(h^A) + \text{Gate}_{C \to B}(h^C)$$

where $h^C$ is the current Region C state (described below). The $C \to B$ feedback signal carries a "crystallization achieved" signal: when Region C has committed to a token, it inhibits further reorganization in Region B.

Region B contains the uncertainty subspace identified by NE-OS Research (2025b). At inference time, a hallucination detector can be applied directly to the intermediate Region B representation without modifying the forward pass.

### 3.4 Region C — Crystal Decoder

Region C consists of $n_C$ transformer layers that receive the output of Region B and a lateral signal from Region A:

$$h^C_0 = h^B + \text{Gate}_{B \to C}(h^B)$$
$$h^C_i = \text{TransformerLayer}_i^C(h^C_{i-1}), \quad i = 1, \ldots, n_C$$

The $B \to C$ gate is defined as:

$$\text{Gate}_{B \to C}(h^B) = -\alpha \cdot W_{B \to C}^{\text{up}} \cdot \sigma(W_{B \to C}^{\text{down}} \cdot h^B)$$

The negative sign and scalar $\alpha > 0$ implement inhibition: Region B suppresses Region C proportionally to its own activation level while it is still organizing. Intuitively, Region B "tells" Region C to wait while semantic structure is still being assembled. As Region B's internal state stabilizes, its activation level decreases and the inhibitory signal weakens, releasing Region C to crystallize.

Region C also includes an early-exit gate: a lightweight confidence probe $p_{\text{conf}} \in [0,1]$ applied at each Region C layer:

$$p_{\text{conf},i} = \sigma(w_{\text{conf}}^T h^C_i)$$

If $p_{\text{conf},i} > \tau$ for threshold $\tau$ (a hyperparameter, default 0.95), the remaining Region C layers are skipped and the current $h^C_i$ is used for output projection. This implements a dynamic depth mechanism analogous to early exit in BERT (Schwartz et al., 2020) but grounded in the observed crystallization phenomenon.

The output logits are:

$$\text{logits} = \text{LM\_head}(\text{RMSNorm}(h^C_{n_C}))$$

### 3.5 Gate Summary

| Gate | Direction | Mechanism | Function |
|------|-----------|-----------|----------|
| $\text{Gate}_{A \to B}$ | A → B | Suppression | A inhibits B dimensions already resolved at low level |
| $\text{Gate}_{B \to C}$ | B → C | Inhibition (negative) | B delays C while semantic organization is incomplete |
| $\text{Gate}_{C \to B}$ | C → B | Feedback | C signals B to halt reorganization once crystallized |

All gates share architecture: a bottleneck linear projection down to $D_g$ dimensions, a sigmoid nonlinearity, and a projection back to $D$ dimensions. Total gate parameters: $3 \times 2 \times D \times D_g$. With $D = 4096$ and $D_g = 64$, this adds approximately 1.6M parameters — negligible relative to a 700M+ model.

### 3.6 Parallel Execution

The key computational advantage of RT-LI is that all three regions can be dispatched in parallel. In a standard transformer with $L$ layers, inference latency is $O(L \cdot T^2 \cdot D / \text{hardware\_throughput})$. In RT-LI, latency is $O(\max(n_A, n_B, n_C) \cdot T^2 \cdot D / \text{hardware\_throughput})$ plus the gate overhead.

Gate signals require brief synchronization points between regions. In practice, the parallel execution follows this schedule:

1. **t=0**: All three regions begin processing simultaneously with their initial inputs ($h^A_0 = h^B_{\text{init}} = h^C_{\text{init}} = \text{Embed}(x)$)
2. **t=k**: After every $k$ sublayers, gate signals are exchanged between regions
3. **t=final**: Region C produces output logits

The synchronization interval $k$ is a hyperparameter controlling the granularity of lateral communication. Setting $k = n_A$ (exchange only at region completion) is a degenerate case equivalent to sequential execution of A → B → C. Setting $k = 1$ maximizes communication but increases synchronization overhead.

### 3.7 Training Objectives

RT-LI is trained with a composite loss:

$$\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda_1 \mathcal{L}_{\text{crystal}} + \lambda_2 \mathcal{L}_{\text{uncertainty}}$$

**Language modeling loss** $\mathcal{L}_{\text{LM}}$: standard next-token prediction cross-entropy applied to Region C output.

**Crystallization loss** $\mathcal{L}_{\text{crystal}}$: encourages Region C to commit early. Applied to the early-exit confidence probe:

$$\mathcal{L}_{\text{crystal}} = -\sum_i p_{\text{conf},i} \cdot \mathbf{1}[\text{Region C}_i \text{ answer is correct}]$$

This teaches the early-exit gate to fire when Region C has already converged to the correct token.

**Uncertainty regularization** $\mathcal{L}_{\text{uncertainty}}$: a supervised signal applied to Region B's intermediate representation to maintain the concentration of uncertainty signal in a low-dimensional subspace. Implemented as a probe loss using held-out calibration data with known correct/incorrect labels.

---

## 4. Theoretical Analysis

### 4.1 Why Parallel Regions Work: The Phase Independence Hypothesis

The central theoretical claim is that the three phases of transformer computation are largely independent conditioned on their inputs — that Phase 1 (chaos encoding) does not require knowledge of what Phase 2 will produce, and similarly for Phase 2 → Phase 3. This is supported by:

1. **Ablation asymmetry**: NE-OS Research (2025a) shows that early layers are not dispensable, but this does not imply that early layers need to be informed by late layers during their computation. The dependency is forward (A is needed by B) not backward (A needs to know about B in advance).

2. **Residual stream as information bus**: In standard transformers, layers communicate through the residual stream. In RT-LI, the gates serve the same function but allow asynchronous communication. The gate from B to C already captures the primary backward-looking dependency: Region C should wait for Region B to stabilize.

3. **Crystallization is a terminal operation**: The abrupt crystallization event (NE-OS Research, 2025a) suggests that Phase 3 layers are doing something qualitatively different from Phase 1 and 2 layers — committing to an output rather than organizing representations. This commitment operation is plausibly parallelizable with the upstream organization.

### 4.2 Capacity Equivalence

A baseline transformer with $L = n_A + n_B + n_C$ layers has the same number of transformer sublayer parameters as RT-LI (assuming uniform hidden dimension). RT-LI adds gate parameters (negligible) and may have different effective capacity due to specialization.

We hypothesize that specialization increases effective capacity: a region trained exclusively on chaos encoding will develop more efficient low-level feature extractors than the same layers in a uniform stack, because they are never penalized for disrupting semantic organization. Empirical validation of this hypothesis is a primary goal of the proposed experiments.

### 4.3 Interpretability by Design

In standard transformers, the uncertainty subspace identified by NE-OS Research (2025b) is an emergent property of L12 in a 28-layer stack — it must be discovered post-hoc through probing. In RT-LI, the uncertainty subspace is structurally localized in Region B by design: Region B is the semantic integrator, and uncertainty about the output token is precisely the degree to which semantic integration is incomplete.

This has a practical consequence: a hallucination detector in RT-LI requires no hyperparameter search for the optimal probe layer. It is always applied to the Region B output, which is architecturally specified as the locus of semantic uncertainty.

### 4.4 Comparison with Mixture of Experts

Mixture of Experts (MoE) architectures (Shazeer et al., 2017; Fedus et al., 2022) also use parallel computation, but at a different granularity: experts are parallel within each feed-forward sublayer, and the layer stack remains sequential. RT-LI differs in two ways:

1. **Granularity**: RT-LI parallelizes across phases (groups of layers), not within layers.
2. **Specialization basis**: MoE experts specialize on input tokens (different tokens route to different experts); RT-LI regions specialize on computational phase (all tokens route to all regions, but regions perform qualitatively different operations).

MoE and RT-LI are orthogonal and could be combined: each RT-LI region could itself be implemented as an MoE feed-forward.

---

## 5. Related Work

**Early exit and adaptive depth.** DeeBERT (Xin et al., 2020) and PABEE (Zhou et al., 2020) add early exit classifiers at intermediate layers of BERT, allowing easy examples to exit early. RT-LI's early exit gate in Region C is conceptually related but grounded in the crystallization mechanism rather than confidence calibration. The key difference is that RT-LI's early exit is within a region specialized for output commitment, not at arbitrary layer boundaries.

**Layer-wise analysis.** Tenney et al. (2019) use edge probing to show that syntactic tasks are solved by early layers and semantic tasks by later layers in BERT. This provides independent evidence for phase-like structure, though without the resolution of the Logit Lens approach used by NE-OS Research (2025a). Geva et al. (2021) show that MLP layers in transformers function as key-value memories, with factual associations concentrated in middle layers — consistent with Phase 2 being the locus of semantic organization.

**Parallel and hierarchical transformers.** Mesh Transformer (Wang and Komatsuzaki, 2021) and related work explore data-parallel training across model depth but do not propose functional specialization. The Perceiver architecture (Jaegle et al., 2021) processes inputs through a latent bottleneck but remains sequential in depth. RT-LI's parallel regions with gate-mediated communication are architecturally novel.

**Neuroscience-inspired architectures.** Predictive coding models (Rao and Ballard, 1999) propose that cortical hierarchies communicate through bottom-up prediction errors and top-down predictions, a form of bidirectional communication analogous to RT-LI's lateral gates. However, predictive coding implementations in deep learning (Millidge et al., 2022) have not achieved competitive performance on language tasks. RT-LI's gates are lighter-weight and grounded in empirical observations about LLM internal representations rather than a full predictive coding framework.

**Hallucination detection.** Kadavath et al. (2022) show that LLMs can express calibrated uncertainty through self-reported probabilities. NE-OS Research (2025b) demonstrates that uncertainty is geometrically concentrated in a low-dimensional subspace of the hidden state before output generation — a stronger claim that enables passive detection without prompting. RT-LI integrates this finding by design.

---

## 6. Proposed Experiments

The following experiments are proposed as future work. We describe each experiment, its methodology, and the expected outcome that would confirm or disconfirm the theoretical claims.

### 6.1 Baseline Comparison: RT-LI 700M vs. Transformer 700M

**Setup.** Train two models from scratch with identical total parameter counts (~700M): (1) a baseline transformer with $L = 28$ uniform layers, (2) an RT-LI with $n_A = 8$, $n_B = 14$, $n_C = 6$ layers and gate dimension $D_g = 64$. Both trained on the same data (C4 or The Pile subset) with the same compute budget (measured in FLOPs). Synchronization interval $k = 2$.

**Evaluation.** Perplexity on WikiText-103 and Lambada (Lawrence, 2001). Zero-shot accuracy on BoolQ, PIQA, HellaSwag.

**Expected outcome.** RT-LI should match transformer baseline perplexity within 2–3% (reflecting comparable capacity) while achieving 2–3× lower inference latency due to parallel region execution. We expect the crystallization loss to reduce average Region C depth used at inference time by 30–40% via early exit.

**Falsification.** If RT-LI perplexity exceeds baseline by more than 5% at matched parameter count and identical compute, the Phase Independence Hypothesis is incorrect and the architecture requires revision — likely by adding more gate synchronization points.

### 6.2 Uncertainty Subspace Localization

**Setup.** Train a linear probe on Region B intermediate representations to predict hallucinations, using the same protocol as NE-OS Research (2025b): prompts with known correct/incorrect completions, probe trained on activations at the Region B midpoint.

**Evaluation.** AUC of hallucination probe on held-out prompts. Dimensionality of the subspace (number of dimensions needed to achieve 95% of full-space AUC).

**Expected outcome.** Probe AUC should match or exceed NE-OS Research (2025b)'s 0.885 (since Region B is explicitly trained to be the locus of semantic uncertainty). Subspace dimensionality should be comparable to the empirically observed 100 dimensions. If RT-LI's architectural inductive bias is working as intended, the probe should require fewer training examples to reach the same AUC, because the uncertainty signal is structurally concentrated rather than emergent.

**Falsification.** If the uncertainty signal in RT-LI is distributed across all three regions rather than concentrated in Region B, the interpretability-by-design claim is false. This would require adding uncertainty regularization loss $\mathcal{L}_{\text{uncertainty}}$ with larger $\lambda_2$.

### 6.3 Lateral Inhibition Gate Ablation

**Setup.** Train four RT-LI variants: (1) full model with all three gates, (2) model without $\text{Gate}_{A \to B}$, (3) model without $\text{Gate}_{B \to C}$, (4) model without $\text{Gate}_{C \to B}$ feedback. All other hyperparameters identical.

**Evaluation.** Perplexity on WikiText-103 and zero-shot accuracy on BoolQ.

**Expected outcome.** Each gate contributes differently. Removing $\text{Gate}_{B \to C}$ (the primary inhibition gate) should cause the largest degradation, because without it Region C cannot be coordinated with Region B's organization phase. Removing $\text{Gate}_{C \to B}$ should cause a smaller but measurable degradation on tasks requiring multi-step reasoning where crystallization can be premature. Removing $\text{Gate}_{A \to B}$ should have minimal effect on perplexity but may increase Region B's effective computation (as measured by activation norms).

### 6.4 Latency Measurement

**Setup.** Measure wall-clock inference latency for RT-LI vs. baseline transformer at batch sizes 1, 8, 32 and sequence lengths 128, 512, 2048, on a single A100 80GB GPU with a PyTorch 2.0 parallel execution implementation.

**Expected outcome.** At batch size 1, latency reduction should be approximately $\frac{n_A + n_B + n_C}{\max(n_A, n_B, n_C)} = \frac{28}{14} = 2\times$ in the ideal case. In practice, we expect 1.5–1.8× due to gate synchronization overhead and memory bandwidth limitations at small batch sizes. At large batch sizes (32+), the speedup should approach 2× as compute becomes the bottleneck rather than memory.

### 6.5 Scaling Behavior of Region B

**Setup.** Train a family of RT-LI models with fixed $n_A = 8$, $n_C = 6$, and varying $n_B \in \{8, 14, 20, 28\}$. Compare against baseline transformers with $L \in \{22, 28, 34, 42\}$ at matched parameter count.

**Expected outcome.** Scaling Region B should yield better perplexity improvement per added parameter than scaling a uniform transformer, because semantic integration is the bottleneck for tasks requiring factual recall and reasoning. If this holds, it provides evidence that the three-region partition correctly identifies the computationally rate-limiting phase.

---

## 7. Limitations

**Gate synchronization overhead.** Parallel execution of regions requires synchronization barriers at each gate exchange. On current hardware, inter-device communication for large activations adds latency. The theoretical 2× speedup may degrade significantly on single-GPU setups with small batch sizes.

**Training instability.** Three-region parallel training with the composite loss introduces additional hyperparameters ($\lambda_1$, $\lambda_2$, $k$, $\tau$, $D_g$, $n_A$, $n_B$, $n_C$). The training dynamics of lateral inhibition gates are untested. It is possible that gates collapse (become zero) or become saturated during training, requiring careful initialization and regularization.

**Phase boundaries are not universal.** NE-OS Research (2025a) observes that the Phase 1/Phase 2 boundary varies from L13.6 to L26.4 depending on the semantic domain. A fixed partition into $n_A = 8$ chaos layers may be suboptimal for tasks where semantic organization begins later. Dynamic partitioning (where region boundaries are learned) is a natural extension but adds architectural complexity.

**Theoretical justification for parallelism.** The Phase Independence Hypothesis is plausible but not proven. If early-layer computations need to be conditioned on late-layer states (e.g., for tasks requiring reasoning backward from a goal), the parallel architecture may fail. The proposed ablation experiments will test this empirically.

**No empirical validation yet.** This paper is a theoretical proposal. All claims about latency, perplexity, and uncertainty localization are predictions, not measurements. The value of this work lies in providing a concrete, falsifiable architecture grounded in empirical observations about transformer internal representations, not in demonstrated performance.

---

## 8. Conclusions

We have proposed RT-LI, a transformer architecture that makes the three-phase structure of LLM computation explicit through parallel specialized regions connected by lateral inhibition gates. The architecture is motivated by two empirical findings from the NE-OS research program: the universal three-phase structure of transformer forward passes (NE-OS Research, 2025a) and the geometric concentration of hallucination-predictive uncertainty in a 100-dimensional subspace at the phase boundary (NE-OS Research, 2025b).

The key architectural innovations are: (1) three parallel regions corresponding to chaos encoding, semantic integration, and crystal decoding; (2) three bidirectional lateral inhibition gates that coordinate suppression and activation across regions; (3) an early-exit gate in Region C grounded in the observed crystallization phenomenon; and (4) a composite training objective that includes a crystallization loss and uncertainty regularization.

Theoretically, RT-LI offers potential advantages in inference latency (parallel execution), interpretability (uncertainty subspace structurally localized in Region B), and modularity (regions can be scaled independently). These predictions are falsifiable through the experimental program described in Section 6.

More broadly, RT-LI represents a methodology for architecture design: use mechanistic interpretability to identify functional structure in trained models, then build architectures whose inductive bias explicitly encodes that structure. If the three-phase structure discovered empirically by NE-OS Research (2025a) is a universal property of autoregressive language modeling rather than an artifact of specific architectures, then RT-LI's inductive bias should accelerate convergence to the same functional structure, requiring less data and compute to learn what the architecture already knows.

---

## References

Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *Journal of Machine Learning Research*, 23(120), 1–39.

Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). Transformer Feed-Forward Layers Are Key-Value Memories. In *Proceedings of EMNLP 2021*.

Hartline, H. K., Wagner, H. G., & Ratliff, F. (1956). Inhibition in the eye of Limulus. *Journal of General Physiology*, 39(5), 651–673.

Jaegle, A., Gimeno, F., Brock, A., Vinyals, O., Zisserman, A., & Carreira, J. (2021). Perceiver: General Perception with Iterative Attention. In *Proceedings of ICML 2021*.

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., ... & Kaplan, J. (2022). Language Models (Mostly) Know What They Know. *arXiv preprint arXiv:2207.05221*.

Michel, P., Levy, O., & Neubig, G. (2019). Are Sixteen Heads Really Better than One? In *Advances in Neural Information Processing Systems 32*.

Millidge, B., Tschantz, A., & Buckley, C. L. (2022). Predictive Coding Approximates Backprop along Arbitrary Computation Graphs. *Neural Computation*, 34(6), 1329–1368.

NE-OS Research. (2025a). Semantic Emergence in Transformer LLMs: Three Phases of Internal Processing. *NE-OS Technical Report*.

NE-OS Research. (2025b). Pre-generation Hallucination Detection via Geometric Probing of the Uncertainty Subspace. *NE-OS Technical Report*.

nostalgebraist. (2020). Interpreting GPT: The Logit Lens. *AI Alignment Forum*. https://www.alignmentforum.org/posts/AcKRB8wDpdaN6v6ru

Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79–87.

Schwartz, R., Stanovsky, G., Swayamdipta, S., Dodge, J., & Smith, N. A. (2020). The Right Tool for the Job: Matching Model and Instance Complexities. In *Proceedings of ACL 2020*.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. In *International Conference on Learning Representations*.

Tenney, I., Das, D., & Pavlick, E. (2019). BERT Rediscovers the Classical NLP Pipeline. In *Proceedings of ACL 2019*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is All You Need. In *Advances in Neural Information Processing Systems 30*.

Wang, B., & Komatsuzaki, A. (2021). GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. *EleutherAI Blog*.

Xin, J., Tang, R., Lee, J., Yu, Y., & Lin, J. (2020). DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference. In *Proceedings of ACL 2020*.

Zhou, W., Xu, C., Ge, T., McAuley, J., Xu, K., & Wei, F. (2020). BERT Loses Patience: Fast and Robust Inference with Early Exit. In *Advances in Neural Information Processing Systems 33*.
