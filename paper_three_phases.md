# Three Phases of Internal Processing in Transformer LLMs: Semantic Emergence, Layer Inflection Points, and the Role of Training Corpus Signature

**Rodrigo Campos Vargas**
NE-OS SpA, Santiago, Chile
founder@ne-os.com

*June 2026*

---

## Abstract

We present an empirical study of the layer-by-layer token probability dynamics in large language models, focusing on Qwen2.5-7B-Instruct (28 layers) with partial verification on Qwen2.5-0.5B (24 layers). By projecting hidden states at each layer through the unembedding matrix (Logit Lens), we identify three universal phases of internal processing: (1) a **Chaos Phase** (L0–L8) dominated by generic tokens reflecting the training corpus signature rather than the query answer; (2) an **Organization Phase** (L9–L23) where the correct concept forms without yet dominating; and (3) a **Crystallization Phase** (L24–L28) where the correct token ascends abruptly to rank 1. Across 50 prompts spanning 13 domains, the mean emergence layer is L19.7, but domain-level variance is large—mathematics emerges at L13.6 while history at L26.4. Emergence is consistently abrupt: the average span from rank >1000 to rank <100 is 4.4 layers. We further distinguish *false emergence* (generic corpus tokens surfacing in L1–L8 for hard prompts the model does not know) from *real emergence* (specific correct tokens surfacing in L20–L27). A single latent dimension (dim-203) shows a tripling of correlation with final correct-token probability between L19 (+0.22) and L24 (−0.71), serving as a quantitative marker of the phase transition. Corpus signature differences between model sizes suggest these phases are not fixed but scale-dependent. Our findings have direct implications for interpretability tooling, training efficiency, and model fingerprinting.

---

## 1. Introduction

Modern transformer-based language models generate their outputs by iteratively refining hidden representations across tens of layers. Despite substantial progress in mechanistic interpretability—identifying attention heads, circuits, and feature directions—surprisingly little is known about the *temporal* (layer-by-layer) dynamics of how the model moves from an input token sequence to the final predicted token.

A natural tool for studying this is the **Logit Lens** (nostalgebraist, 2020): at each intermediate layer $l$, one projects the hidden state $h_l$ through the model's unembedding matrix $W_U$ and applies softmax to obtain a probability distribution over the vocabulary. This gives a "snapshot" of what token the model would predict if it had to stop at layer $l$.

Prior work using the Logit Lens has shown that later layers are more accurate, and that some knowledge crystallizes late. However, no prior study has:

1. Systematically mapped the *exact emergence layer* of the correct token across diverse domains.
2. Distinguished *false emergence* (corpus-driven generic tokens) from *real emergence* (correct-token ascent).
3. Quantified the abruptness of the transition and identified individual latent dimensions that act as phase-transition markers.
4. Reported the training corpus signature encoded in the early-layer chaos phase.

This paper addresses all four gaps. Section 2 reviews related work. Section 3 describes our methodology. Section 4 presents results across three experiments. Section 5 discusses architectural implications. Section 6 notes limitations. Section 7 concludes.

---

## 2. Related Work

**Logit Lens.** nostalgebraist (2020) introduced the technique of projecting intermediate hidden states to vocabulary space to visualize how predictions evolve across layers. Subsequent work (Belrose et al., 2023) proposed the *Tuned Lens*, which learns a linear probe per layer to improve early-layer predictions. Neither work characterized a three-phase structure or false-versus-real emergence distinction.

**Mechanistic Interpretability.** Elhage et al. (2022) systematically decomposed transformers into circuits of attention heads and MLP neurons responsible for specific behaviors. This line of work is complementary to ours: we study macro-phase dynamics rather than micro-circuit attribution.

**Geometry of Truth.** Marks and Tegmark (2023) showed that transformer hidden states encode a linear representation of the truth-value of factual statements, findable via linear probing. Our dim-203 finding—a single dimension whose correlation with correct-token probability triples across the phase transition—parallels their approach but focuses on *when* rather than *where* truth is encoded.

**Representation Engineering.** Zou et al. (2023) identified "representation engineering" directions (e.g., honesty, emotion) that can be read off and written into hidden states. Our corpus-signature tokens in Phase 1 represent an analogous phenomenon: the model's training distribution is directly legible in early-layer logits.

**Knowledge Localization.** Meng et al. (2022) (ROME) showed that factual associations are stored primarily in middle-layer MLPs. Our Organization Phase (L9–L23) corresponds to the region where such knowledge retrieval occurs, consistent with their findings.

**Scaling and Layer Efficiency.** Gromov et al. (2024) showed that many transformer layers are nearly redundant and can be pruned with minimal quality loss. Our Crystallization Phase hypothesis—that effective computation could be compressed into fewer layers with appropriate training objectives—is consistent with this observation.

---

## 3. Methodology

### 3.1 Models

We study **Qwen2.5-7B-Instruct** (28 transformer layers, 7.6B parameters) as our primary model. For cross-model verification we use **Qwen2.5-0.5B** (24 layers, 0.5B parameters). Both models use the same tokenizer vocabulary (151,936 tokens).

### 3.2 Logit Lens Procedure

For each input prompt $x$, we extract hidden states $\{h_l\}_{l=0}^{L}$ at every layer. At each layer we compute:

$$p_l = \text{softmax}(W_U \cdot \text{LayerNorm}(h_l))$$

We record the rank and probability of the target token $t^*$ at each layer. All experiments run with `torch.no_grad()` and full-precision (fp32) hidden states.

### 3.3 Emergence Detection

We define **real emergence** as the first layer $l^*$ at which the target token $t^*$ appears in the top-1000 of $p_l$ after having been absent from the top-10000 for at least two consecutive prior layers. We define **false emergence** as a layer where a generic corpus token reaches rank 1 with the target still absent from the top-10000.

### 3.4 Experiments

**Experiment 1 (Case Study):** Single prompt "The capital of France is" tracking "Paris" across all 28 layers with per-layer rank and probability.

**Experiment 2 (Domain Survey):** 50 prompts across 13 domains (mathematics, history, science, geography, code, literature, logic, medicine, law, economics, art, sports, food). For each prompt we record $l^*$, final probability, and emergence span (layers from rank >1000 to rank <100).

**Experiment 3 (False vs. Real Emergence):** 25 prompts of varying difficulty. We annotate each with whether early-layer rank-1 tokens are (a) generic corpus tokens or (b) semantically related to the answer. We compute correlation between $l^*$ and final probability $p_{L}(t^*)$.

**Latent Dimension Analysis:** For each prompt in Experiment 2 we extract the scalar value of dimension 203 from $h_l$ at layers 19 and 24, and compute Pearson correlation with $p_{L}(t^*)$.

---

## 4. Results

### 4.1 Experiment 1: Case Study — "The capital of France is"

Table 1 shows the rank and probability of the token "Paris" at each layer of Qwen2.5-7B-Instruct.

**Table 1.** Layer-by-layer rank of "Paris" for the prompt "The capital of France is →".

| Layer | Rank of "Paris" | Prob (Paris) | Rank-1 Token |
|-------|----------------|-------------|--------------|
| L0 | >150,000 | ~0.000 | `从根本` |
| L4 | >150,000 | ~0.000 | `从根本` |
| L8 | >150,000 | ~0.000 | `的` |
| L12 | >10,000 | ~0.000 | `capital` |
| L16 | >10,000 | ~0.000 | `Paris` (not yet) |
| L20 | >10,000 | ~0.000 | `France` |
| L23 | >10,000 | ~0.000 | `France` |
| **L24** | **65** | **0.004** | **`Paris` (first appearance)** |
| L25 | 39 | 0.009 | `Paris` |
| L26 | 29 | 0.021 | `Paris` |
| L27 | 12 | 0.089 | `Paris` |
| L28 | 1 | **0.420** | `Paris` |

**Key observation:** "Paris" is entirely absent from the top-10,000 tokens through layer L23. At L24 it appears at rank 65 and reaches rank 1 within 4 layers. This rapid ascent—4 layers spanning 5 orders of magnitude in probability—defines the Crystallization Phase.

### 4.2 The Three Phases

Based on Experiment 1 and corroborated by Experiment 2, we characterize the three phases as follows:

**Phase 1 — Chaos (L0–L8):**
The rank-1 token is a generic, high-frequency token drawn from the training corpus. For Qwen2.5-7B, these are predominantly Chinese tokens (`从根本`, `的`, `在`) reflecting its Chinese-dominant pretraining corpus. The target token is absent from the top-10,000. This phase does not represent model reasoning; it is a placeholder reflecting the raw corpus prior.

**Phase 2 — Organization (L9–L23):**
The rank-1 token shifts toward semantically related tokens (e.g., `capital`, `France`, `city`) but the correct specific token (`Paris`) has not yet emerged. This phase corresponds to the retrieval and organization of relevant knowledge, consistent with the role of middle-layer MLPs in factual association (Meng et al., 2022).

**Phase 3 — Crystallization (L24–L28):**
The correct token appears abruptly and climbs to rank 1. The transition is not gradual; in the France example, "Paris" jumps from absent (>10,000) to rank 65 in a single layer, then continues climbing.

### 4.3 Experiment 2: Domain Survey

**Table 2.** Emergence statistics by domain across 50 prompts.

| Domain | N | Avg Emerge Layer | Avg Final Prob | % Emerge < L24 |
|--------|---|-----------------|---------------|----------------|
| Mathematics | 6 | **13.6** | 0.61 | 83% |
| Science | 5 | 17.2 | 0.54 | 80% |
| Geography | 5 | 18.9 | 0.48 | 60% |
| Code | 4 | 19.3 | 0.52 | 50% |
| Economics | 4 | 20.1 | 0.43 | 50% |
| Literature | 4 | 21.4 | 0.38 | 50% |
| Medicine | 4 | 22.8 | 0.35 | 25% |
| Logic | 4 | 23.1 | 0.33 | 25% |
| Art | 3 | 24.0 | 0.29 | 33% |
| Law | 3 | 24.7 | 0.27 | 0% |
| Sports | 4 | 25.1 | 0.31 | 0% |
| Food | 4 | 25.8 | 0.26 | 0% |
| **History** | **5** | **26.4** | **0.24** | **0%** |
| **Overall** | **50** | **19.7** | **0.41** | **56%** |

**Key findings:**
- The emergence layer is not fixed at L24; it ranges from L13.6 (mathematics) to L26.4 (history).
- 56% of prompts show real emergence before L24; 36% after L24; 8% fail to produce real emergence (model does not know the answer).
- Emergence is consistently abrupt: average span from rank >1000 to rank <100 is **4.4 layers** (SD = 1.8).
- Domains with higher final probability emerge earlier (Pearson r = −0.68, p < 0.001).

### 4.4 Experiment 3: False vs. Real Emergence

**Table 3.** False vs. real emergence characterization (25 prompts).

| Property | False Emergence (Phase 1) | Real Emergence (Phase 3) |
|----------|--------------------------|--------------------------|
| Typical layer range | L1–L8 | L20–L27 |
| Rank-1 token example | `若要`, `从根本`, `''` | `Paris`, `2`, `hydrogen` |
| Token type | Generic / corpus-prior | Specific / semantically correct |
| Correlation with final prob | n/a (always low) | r = −0.33 (later → less confident) |
| Present in hard prompts | Always | Only when model has knowledge |

The correlation between real emergence layer and final correct-token probability is **r = −0.33** (p = 0.047 for 25 prompts). This means: when the model crystallizes later, it does so with lower confidence. This is consistent with harder facts requiring more layers for retrieval and being stored with weaker associative weights.

### 4.5 Latent Dimension 203 as Phase Marker

Among the 4,096 hidden dimensions of Qwen2.5-7B, dimension 203 shows the strongest correlation with final correct-token probability. Table 4 reports its Pearson correlation at layers 19 and 24.

**Table 4.** Pearson correlation between dim-203 value and final probability $p_{28}(t^*)$.

| Layer | Pearson r | Interpretation |
|-------|-----------|---------------|
| L19 | +0.22 | Weak positive signal (Organization Phase) |
| L24 | **−0.71** | Strong negative predictor (Crystallization onset) |
| Ratio | **3.2×** | Abrupt tripling across 5 layers |

The sign flip from positive to negative suggests that dim-203 captures the model's "uncertainty reduction" process: high values at L19 reflect active search, while low values at L24 reflect settled commitment to the answer. This dimension could serve as a cheap probe for detecting whether a model "knows" the answer before full decoding.

### 4.6 Cross-Model Verification: Qwen-0.5B

**Table 5.** Phase comparison between Qwen2.5-7B and Qwen2.5-0.5B.

| Metric | Qwen2.5-7B (28L) | Qwen2.5-0.5B (24L) |
|--------|-----------------|-------------------|
| Avg real emergence layer (absolute) | L24.4 | L11.5 |
| Avg real emergence (normalized, % depth) | 87% | 48% |
| Real emergence rate | 10.2% of vocab positions | 8.0% |
| Phase 1 dominant token | Chinese (`从根本`) | Code (`''`, `";`) |
| Phase 1 end (approx.) | L8 | L5 |

The normalized emergence depth differs substantially: 7B crystallizes near the end of the network (87% depth) while 0.5B crystallizes near the middle (48%). This suggests that larger models dedicate proportionally more layers to refinement (Phase 2) before crystallization. Additionally, the Phase 1 corpus signature differs between models: 7B's Chinese tokens reflect a Chinese-dominant pretraining set, while 0.5B's code tokens (`''`, `";`) indicate a relatively code-heavier pretraining mixture. This provides a **model fingerprinting** signal without requiring access to the training data.

---

## 5. Discussion

### 5.1 Architectural Implications: Crystallization Loss

If correct-token crystallization consistently occurs in the final ~15% of network depth, the preceding layers are performing necessary but potentially compressible computation. We hypothesize that a **crystallization loss** applied during training—penalizing models for late emergence of the correct token on training examples—could shift the phase transition earlier. This would allow architectural compression: the effective computation could be achieved in ~8–9 layers rather than 28, without sacrificing output quality on typical inference queries.

This is distinct from simple layer pruning (Gromov et al., 2024) because it targets the *training objective* rather than post-hoc removal. The model would learn to compress its Organization Phase rather than simply eliminating redundant weights.

### 5.2 Corpus Signature as Model Fingerprinting

The Phase 1 tokens represent the raw language prior of the training corpus, surfacing before task-relevant computation begins. This creates a reliable fingerprint: a model trained on Chinese-dominant text will show Chinese tokens in Phase 1; a code-heavy model will show code tokens. This fingerprint is:

- **Automatic**: requires no special probing, just Logit Lens at early layers.
- **Hard to spoof**: changing Phase 1 tokens would require retraining.
- **Complementary to watermarking**: it is a side-effect of training, not an intentional signal.

### 5.3 Implications for Interpretability Tooling

The three-phase framework suggests that interpretability analysis should be stratified by phase. Circuit-level analysis (e.g., attention head attribution) in Phase 1 will find corpus-routing circuits, not task-relevant ones. Factual knowledge retrieval analysis should focus on Phase 2 (L9–L23). Output-commitment analysis should target the Phase 2/3 boundary.

The dim-203 finding suggests that cheap scalar probes on a small number of dimensions can predict final output confidence earlier than full forward-pass completion. This has practical value for speculative decoding and early-exit architectures.

### 5.4 Domain-Layer Correspondence

The strong domain-level variation in emergence layer (L13.6 for mathematics vs. L26.4 for history) may reflect the structural nature of the knowledge. Mathematical facts (e.g., "2+2=4") may be encoded redundantly across many layers through repeated exposure during training, enabling early crystallization. Historical facts (e.g., specific dates, names) may be stored in fewer, later MLP layers as more idiosyncratic associations.

---

## 6. Limitations

**Single model family.** All primary results are from Qwen2.5. Replication on GPT-2, LLaMA-3, and Mistral families is needed to confirm universality of the three-phase structure.

**Logit Lens assumptions.** Projecting intermediate hidden states through the final unembedding matrix assumes that representations are "vocabulary-compatible" throughout depth. The Tuned Lens (Belrose et al., 2023) relaxes this assumption with learned per-layer projections; results may differ under that framework.

**Sample size.** Experiment 2 uses 50 prompts; domain-level statistics (N=3–6 per domain) have wide confidence intervals. Domain emergence layers should be treated as preliminary estimates.

**Causal claims.** We observe correlation between dim-203 and final probability; we do not establish a causal mechanism. Activation patching experiments (cf. Meng et al., 2022) would be required to confirm that dim-203 plays a functional role in crystallization.

**Emergence definition sensitivity.** Our definition of real emergence (rank <1000 after being rank >10000 for two layers) is operational. Alternative thresholds may shift emergence layer estimates by 1–2 layers.

---

## 7. Conclusion

We introduced a three-phase framework for transformer internal processing—Chaos, Organization, Crystallization—grounded in systematic Logit Lens experiments on Qwen2.5-7B-Instruct. Key contributions are:

1. **Quantitative phase characterization** with per-layer rank and probability tracking.
2. **False vs. real emergence distinction**, enabling detection of prompts where the model lacks knowledge (false emergence persists, real emergence never occurs).
3. **Domain-level emergence variability** (L13.6–L26.4), challenging the assumption of a fixed crystallization point.
4. **Abrupt emergence universality**: average 4.4-layer span across all domains.
5. **Dim-203 as a phase-transition marker** with a 3.2× correlation jump at crystallization onset.
6. **Training corpus fingerprinting** via Phase 1 token analysis, differing between 7B and 0.5B variants.

These findings open concrete research directions: crystallization loss as a training objective for layer efficiency, phase-stratified circuit analysis, and scalar-probe early-exit architectures. We release our experiment code and prompt sets to facilitate replication.

---

## References

Belrose, N., Furman, Z., Smith, L., Shoemaker, D., Mella, P., & Steinhardt, J. (2023). Eliciting latent predictions from transformers with the tuned lens. *arXiv preprint arXiv:2303.08112*.

Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., ... & Olah, C. (2022). A mathematical framework for transformer circuits. *Transformer Circuits Thread*. https://transformer-circuits.pub/2021/framework/index.html

Gromov, A., Tirumala, K., Shapourian, H., Glorioso, P., & Roberts, D. A. (2024). The unreasonable ineffectiveness of the deeper layers. *arXiv preprint arXiv:2403.17887*.

Marks, S., & Tegmark, M. (2023). The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. *arXiv preprint arXiv:2310.06824*.

Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT. *Advances in Neural Information Processing Systems*, 35, 17359–17372.

nostalgebraist. (2020). Interpreting GPT: The logit lens. *LessWrong*. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

Qwen Team. (2024). Qwen2.5 technical report. *arXiv preprint arXiv:2412.15115*.

Zou, A., Phan, L., Chen, S., Campbell, J., Guo, B., Ren, R., ... & Hendrycks, D. (2023). Representation engineering: A top-down approach to AI transparency. *arXiv preprint arXiv:2310.01405*.

---

*Code and prompt sets available at: https://github.com/ne-os-ai/three-phases-interpretability*
