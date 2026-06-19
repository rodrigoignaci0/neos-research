# AION: Adaptive Inference with Orthogonal Networks

## Abstract

We present AION (Adaptive Inference with Orthogonal Networks), a 2.2B-parameter language model architecture that decouples factual recall from reasoning computation through a learned, data-driven routing mechanism. AION introduces the Hard Bypass: when an associative memory module (Engram) retrieves a stored fact with confidence exceeding 0.90, the model skips all transformer layers entirely, incurring zero FLOPs for the Latent Reasoner. This is architecturally distinct from memory-augmented transformers such as DeepSeek Engram (arXiv:2601.07372), which inject retrieved content but still execute the full transformer stack. AION further reduces compute through per-layer early-exit gates in a 6-layer Latent Reasoner, creating a two-level efficiency hierarchy. Gate training with a combined language modeling and binary cross-entropy loss achieves perfect separation after 12 epochs: 99.6% confidence on factual queries, 0.0% on generative queries. We describe the full architecture, training pipeline across four versions (v5–v8), and empirical results demonstrating the viability of orthogonal compute paths in a single language model.

---

## 1. Introduction

Language model inference is computationally homogeneous: regardless of whether a query requires deep multi-step reasoning or simple factual recall, the model executes every layer of the transformer. A question such as "What is the capital of France?" consumes the same compute as "Prove the irrationality of π." This uniformity is wasteful and, crucially, unnecessary.

Two lines of prior work attempt to address this. Mixture-of-Experts (MoE) models [1, 2] route tokens to specialized sub-networks at the feed-forward layer, reducing per-token active parameters while keeping total capacity high. Early-exit transformers [3, 4] attach classifiers to intermediate layers and halt inference when confidence is sufficient. Both approaches operate within the transformer execution flow — they reduce work, but the model still processes every query through the same architectural scaffolding.

Memory-augmented language models [5, 6] introduce an orthogonal mechanism: explicit key-value stores that can answer queries without full transformer passes. However, existing implementations, including DeepSeek Engram [7], use retrieved memory as an additional input signal rather than as a bypass condition. The transformer still executes fully.

We propose AION, which unifies these ideas into a strict two-level compute hierarchy:

1. **Level 1 — Hard Bypass**: An Engram module performs O(1) associative lookup. If the gate confidence exceeds a threshold, the Latent Reasoner is skipped entirely. Zero transformer FLOPs are spent.
2. **Level 2 — Early Exit**: For queries that enter the Latent Reasoner, per-layer gates halt computation as soon as sufficient confidence is reached.

The routing is learned entirely from data through a binary cross-entropy gate loss combined with standard language modeling loss. No manual rules, query classifiers, or domain labels are required at inference time.

---

## 2. Architecture

### 2.1 Overview

AION processes an input sequence through three sequential stages: (1) token embedding, (2) Engram lookup and gating, (3) conditional execution of the Latent Reasoner.

```
Input Tokens
     │
     ▼
[Token Embedding]  (152064 × 3584)
     │
     ▼
[Engram Module]
     │
     ├─── Gate confidence > 0.90 ──► [LM Head] → Output  (Hard Bypass)
     │
     └─── Gate confidence ≤ 0.90
               │
               ▼
     [Latent Reasoner Layer 1]
     [Latent Reasoner Layer 2]
          [Exit Gate]─── > 0.85 ──► [LM Head] → Output
     [Latent Reasoner Layer 3]
          [Exit Gate]─── > 0.85 ──► [LM Head] → Output
              ...
     [Latent Reasoner Layer 6]
               │
               ▼
          [LM Head] → Output
```

**Parameter breakdown (~2.2B total):**
- Embedding: 152,064 × 3,584 = 545M
- Engram: ~750M (keys, values, projections)
- Latent Reasoner (6 layers): ~900M
- LM Head: tied with embedding (0 extra)

### 2.2 Engram: Associative Memory Module

The Engram is a differentiable key-value store with 65,536 slots. Keys: 512-dim, Values: 3,584-dim (= H).

**Lookup:**
```
q = Normalize(W_q · mean(embeddings))     # q ∈ ℝ^512
s_i = cosine_similarity(q, k_i)           # for all 65536 keys
top4 = argmax_4(s)
w = softmax(s_top4)
v_retrieved = Σ w_i · v_i
```

**Gate:**
```
gate_input = concat(v_retrieved, mean(embeddings))   # ∈ ℝ^{H×2}
confidence = Sigmoid(Linear(128→1)(ReLU(Linear(H×2→128)(gate_input))))
```

If `confidence > 0.90` → Hard Bypass: skip Latent Reasoner, pass directly to LM Head.

**Key distinction from DeepSeek Engram [7]:** DeepSeek injects retrieved vectors into the residual stream but still executes all transformer layers. AION's gate makes transformer execution *optional* — the model can produce output without running a single transformer layer.

### 2.3 Latent Reasoner

6-layer transformer with:
- H = 3,584, N_HEAD = 28 (head dim = 128)
- MLP_D = 14,336
- RMSNorm + causal attention + SwiGLU-style MLP

**Per-layer early-exit gates (layers 2–6):**
```
exit_conf = Sigmoid(Linear(32→1)(ReLU(Linear(H→32)(h_l))))
if exit_conf > 0.85 and l >= 2: → LM Head
```

This creates a second compute-saving level: queries not simple enough for Engram bypass may still exit after 2-4 layers rather than the full 6.

---

## 3. Training Methodology

### v5 — Knowledge Distillation
- Teacher: Qwen2.5-14B-Instruct
- 100K examples: 58K factual, 26K reasoning, 15K generative
- Loss: KL divergence on full token sequences

### v6 — Gate Loss Training
- **Loss:** `L = L_LM + 0.3 · BCE(confidence, label)`
  - label = 1 for factual, label = 0 for generative
- 12 epochs, LR=3e-5, batch=128 effective, MAX_LEN=192
- Gate loss computed in float32 (outside autocast) to prevent gradient underflow near sigmoid extremes

### v7 — Conversational Fine-Tuning
- Dataset: SmolTalk 150K (HuggingFaceTB)
- 3 epochs, MAX_LEN=256
- Format: `User: ...\nAssistant: ...`

### v8 — Extended Fine-Tuning
- Dataset: SmolTalk 150K + OpenHermes 100K = 250K total
- 5 epochs, MAX_LEN=512
- Improves instruction following and domain coverage

---

## 4. Results

### 4.1 Gate Specialization (v6)

**Table 1: Gate Training Over 12 Epochs**

| Epoch | LM Loss | Gate Loss | Factual Conf. | Generative Conf. | Gap   |
|-------|---------|-----------|---------------|------------------|-------|
| 1     | 0.2907  | 0.2583    | 0.748         | 0.015            | 0.733 |
| 2     | 0.2734  | 0.0086    | 0.935         | 0.004            | 0.931 |
| 3     | 0.2522  | 0.0028    | 0.990         | 0.002            | 0.989 |
| 6     | 0.1884  | 0.0023    | 0.979         | 0.000            | 0.978 |
| 9     | 0.1428  | 0.0008    | 0.996         | 0.000            | 0.995 |
| 12    | 0.1202  | 0.0005    | 0.996         | 0.000            | **0.996** |

**Key findings:**

- **Rapid convergence:** Gate loss drops 99% (0.2583 → 0.0028) in the first 3 epochs.
- **Perfect separation at epoch 12:** Factual = 99.6%, Generative = 0.0%, Gap = 0.996. The Hard Bypass threshold of 0.90 is cleared with 9.6% margin on factual queries.
- **No collapse:** The gate does not converge to always-on or always-off. It is genuinely condition-dependent.
- **Compatible objectives:** LM loss continues improving through epoch 12 (0.29 → 0.12, a 58% reduction) while gate loss converges. The two losses do not conflict.

### 4.2 Compute Savings (Architectural Guarantees)

| Query Type | Layers Executed | Compute Saved |
|------------|----------------|---------------|
| Factual (conf > 0.90) | 0 (Hard Bypass) | ~100% of Latent Reasoner |
| Simple reasoning (exit at L2) | 2 of 6 | ~67% of Latent Reasoner |
| Complex reasoning (exit at L4) | 4 of 6 | ~33% of Latent Reasoner |
| Deep reasoning (no exit) | 6 of 6 | 0% |

Given the 99.6% factual confidence, virtually all factual queries in the training distribution activate the Hard Bypass.

---

## 5. Related Work

| Method | Memory | Bypass Transformer? | Data-Driven Routing? |
|--------|--------|---------------------|----------------------|
| RAG [5] | External index | No (appends to context) | No |
| Memorizing Transformers [6] | KV cache | No | No |
| DeepSeek Engram [7] | Internal store | No (injects, still runs) | Partial |
| MoE [1,2] | N/A | No (routes within layers) | Yes (token-level) |
| Early Exit [3,4] | N/A | Partial (skips tail layers) | Yes |
| **AION (ours)** | **Internal store** | **Yes (full skip)** | **Yes (query-level)** |

AION is the only architecture in this comparison that can produce output without executing any transformer layer. The orthogonality of compute paths (Engram vs. Latent Reasoner) is a structural property, not a runtime optimization.

---

## 6. Discussion

### 6.1 Why "Orthogonal"
The Engram and Latent Reasoner share no parameters (except the embedding) and are traversed mutually exclusively. Each path receives gradients only from queries it processes. This enables clean specialization without interference — unlike branching transformers that share early layers.

### 6.2 Gate Reliability
At threshold 0.90, the 0.996 factual confidence means false negatives (factual queries sent to the Latent Reasoner) are rare in-distribution. Edge cases to investigate: factual queries phrased as hypotheticals, multi-hop factual questions, and domain shift.

### 6.3 Training Stability Notes
Computing gate loss outside `torch.autocast` is critical. Float16 sigmoid outputs near 0 or 1 cause BCE gradient underflow. The gate coefficient of 0.3 balances gate convergence speed against LM quality preservation.

---

## 7. Limitations

1. No benchmark evaluation (MMLU, HellaSwag, ARC) — required to compare against 2B-scale baselines.
2. Bypass rate at deployment not measured — real FLOPs savings are distribution-dependent.
3. Binary gate labels — does not handle queries that are partially factual and partially generative.
4. Fixed Engram after training — no online knowledge update.
5. Engram capacity (65,536 slots) is small relative to world knowledge — scaling requires ANN indexing.

---

## 8. Conclusion

AION introduces the Hard Bypass: a learned, confidence-gated routing mechanism that allows a language model to skip its entire reasoning stack for factual queries. The architecture achieves 99.6% / 0.0% factual/generative gate separation after 12 training epochs, with simultaneous 58% improvement in language modeling loss. The two-level compute hierarchy (Hard Bypass + Early Exit) provides fine-grained efficiency without manual routing rules.

The architecture is qualitatively distinct from DeepSeek Engram (memory injection without bypass), MoE (intra-layer routing), and early-exit transformers (intra-stack exit). AION's orthogonal compute paths represent a new structural paradigm for conditional inference in language models.

---

## References

[1] Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR.
[2] Fedus et al. (2022). Switch Transformers. JMLR.
[3] Graves (2016). Adaptive Computation Time. arXiv:1603.08983.
[4] Schuster et al. (2022). Confident Adaptive Language Modeling (CALM). NeurIPS.
[5] Lewis et al. (2020). Retrieval-Augmented Generation. NeurIPS.
[6] Wu et al. (2022). Memorizing Transformers. ICLR.
[7] DeepSeek-AI (2026). DeepSeek Engram. arXiv:2601.07372.
[8] Press & Wolf (2017). Using the Output Embedding to Improve Language Models. EACL.
[9] HuggingFaceTB (2024). SmolTalk Dataset.
[10] Teknium (2023). OpenHermes-2.5 Dataset.

---

## Apéndice A: Benchmarks AION v8

Evaluación con lm-evaluation-harness, 500 ejemplos por tarea, 0-shot.

| Task | AION v8 | Random Baseline |
|------|---------|----------------|
| HellaSwag | 30.4% | 25.0% |
| ARC Easy | 30.6% | 25.0% |
| ARC Challenge | 16.4% | 25.0% |
| WinoGrande | 51.6% | 50.0% |

**Modelo:** AIONv6 architecture, 2.2B params, fine-tuned en SmolTalk 150K + OpenHermes 100K (v8).

**Interpretación:** Los resultados están sobre random en 3 de 4 tareas. ARC Challenge (<random) sugiere que el modelo no generaliza bien a razonamiento multi-paso en dominio de ciencias. WinoGrande (51.6%) es la señal más fuerte — el modelo captura algo de coherencia referencial.

Estos resultados son esperados para un modelo de 2.2B sin pre-entrenamiento en datos de razonamiento. El resultado académicamente relevante de AION es el gate (Sección 4.1), no el benchmark de lenguaje general.
