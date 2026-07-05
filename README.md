# NE-OS Research Papers

Independent AI research by **Rodrigo Campos Vargas** — [NE-OS SpA](https://ne-os.com), Chile

---

## Papers (2026)

### [Hallucination as Geometric Collapse in the Residual Stream](./paper_hallucination_geometry.md)
> Pre-generation hallucination detection from hidden states. AUC 0.885 from 100 of 4096 dimensions (41× compression). LOCO-CV shows probes learn domain identity, not uncertainty — AUC drops 0.885→0.556 on held-out categories.

**Status:** Submitted — BlackboxNLP 2026 (ACL)
**OpenReview:** [openreview.net/forum?id=CPo8Gn7Mi9](https://openreview.net/forum?id=CPo8Gn7Mi9)

---

### [RT-LI: Regional Transformer with Lateral Inhibition](./paper_rtli.md)
> Architecture partitioning transformer computation into 3 parallel specialized regions (Chaos Encoder, Semantic Integrator, Crystal Decoder) connected by lateral inhibition gates. Based on empirically discovered three-phase structure confirmed across Qwen2.5-7B, Mistral-7B, Gemma-2B. Trained 258M-param model: Region C crystallizes correctly (relativity 95.2%, oxygen 71.9%), all inter-region gates active.

**Status:** Experiments running — July 2026

---

### [DAMA: Distributed Associative Memory in LLMs](./paper_dama.md)
> Hypothesis: the 100-dim uncertainty subspace decomposes into M specialized agents encoding domain-specific uncertainty. ICA experiment shows emergent domain specialization without supervision (geography AUC 0.913 from agent_4 alone).

**Status:** Experiments complete — July 2026

---

### [ARIA: Adaptive Regions with Internal Awareness](./paper_aria.md)
> Full architecture: RT-LI + recurrent B↔C feedback + DAMA memory (8 agents × 64 dims) + Test-Time Training + curriculum learning + JEPA loss + contrastive consistency. 258M params.

**Status:** Architecture complete, training pending RT-LI validation — July 2026

---

### [Three Phases of Internal Processing in Transformer LLMs](./paper_three_phases.md)
> Universal 3-phase model: Chaos (L0–8) → Semantic Organization (L9–23) → Crystallization (L24+). Confirmed across Qwen, Mistral, Gemma. Average real emergence at L19.7/28.

**Status:** Complete

---

### [Universal Latent Space Alignment Across LLM Families](./paper_latent_alignment.md)
> Empirical confirmation of Platonic Representation Hypothesis. Mistral/Falcon/Gemma → Qwen cos > 0.99 via 4MB linear projection.

**DOI:** [10.5281/zenodo.20757527](https://zenodo.org/record/20757527)

---

### [AION: Adaptive Inference with Orthogonal Networks](./aion_paper.md)
> 2.2B-parameter LLM with Hard Bypass: skips all transformer layers for factual queries. 99.6%/0.0% gate separation.

**DOI:** [10.5281/zenodo.20757442](https://zenodo.org/record/20757442)

---

### [RCLA: Resonance-Coded Language Architecture](./rcla_paper.md)
> Replaces transformer feed-forward layers with sparse template routing. 13.4% lower loss than transformer baseline, 1.61x faster training.

**DOI:** [10.5281/zenodo.20757444](https://zenodo.org/record/20757444)

---

## Research Focus

All papers investigate the **internal geometry of transformer LLMs** — how information is organized, compressed, and retrieved across layers, and how this geometry can be exploited to build better architectures and detect failure modes before they manifest as output tokens.

Core empirical finding: transformers universally exhibit three functional phases, regardless of architecture family, scale, or training data. RT-LI and ARIA make this structure architecturally explicit.

## Contact

**Rodrigo Campos Vargas** — NE-OS SpA, Santiago, Chile
GitHub: [@rodrigoignaci0](https://github.com/rodrigoignaci0)
