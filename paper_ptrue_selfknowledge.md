# Instruction Tuning Improves P(True) Discrimination but Not Calibration: Evidence for a Pretraining Data Confound

**Rodrigo Campos Vargas**
NE-OS SpA, Santiago, Chile
rodrigo@ne-os.com

---

## Abstract

P(True) — the probability a language model assigns to an affirmative token when asked whether its own answer is correct — has been proposed as a measure of self-knowledge. Kadavath et al. (2022) found that instruction-tuned models exhibit higher P(True) quality. We revisit this claim across four base/instruct model families (Llama3-8B, Gemma2-2B, Gemma2-9B, Qwen2.5-7B) using domain-stratified leave-one-domain-out (LOCO) AUC, item-level accuracy matching, and bootstrap confidence intervals with Bonferroni correction. Discrimination improves in point estimate in 3 of 4 families but is statistically significant in only 2 (Gemma2-2B +0.234, 98.75% CI [+0.160, +0.304]; Gemma2-9B +0.115, [+0.046, +0.182]); Llama3-8B is positive but not significant (+0.066, [−0.008, +0.137]) and Qwen2.5-7B is null-to-negative (−0.030, [−0.068, +0.013]), consistent with pretraining saturation. Calibration moves the opposite way: ECE worsens after instruction tuning in 3 of 4 families on TriviaQA and in all 4 on SciQ. The instruct advantage is also format-dependent: chat templates reduce LOCO AUC for all four instruct models and reverse the base/instruct direction for Llama. Critically, OLMo-2-7B without any alignment achieves LOCO AUC of 0.773 — within 0.015 of Llama3-8B-base and inside the range of aligned models — while Pythia-6.9B, trained on a less QA-rich corpus, reaches only 0.491; the highest LOCO in our suite belongs to an unaligned model (Qwen2.5-7B-base, 0.827). Together these results indicate that pretraining data composition is a sufficient driver of P(True) discrimination, while alignment tends to degrade calibration. We recommend reporting both AUC and ECE, using alias-based answer scoring, and applying item-level matching.

---

## 1. Introduction

Language models are increasingly deployed in settings where reliable self-assessment is critical. A model that can accurately identify when its own answers are correct would enable downstream applications including abstention, retrieval augmentation triggers, and confidence-gated pipelines. Kadavath et al. (2022) introduced P(True) as an operationalization of this capacity: given a question and a sampled answer, the model is asked whether the answer is correct, and the probability assigned to "True" is treated as a confidence score. They found that instruction-tuned models exhibit higher P(True) quality, suggesting alignment improves self-knowledge.

This finding, if robust, would have significant implications: self-assessment as an emergent property of alignment, improvable by instruction tuning. However, the claim has not been systematically tested with (1) controlled item-level matching to eliminate accuracy confounds, (2) proper alias-based answer scoring, (3) ECE measurement alongside AUC, or (4) pretraining controls to isolate alignment effects.

We address these gaps. Our results are mixed: the AUC improvement replicates directionally in 3 of 4 families but is statistically significant in only 2 after Bonferroni correction, and one family (Qwen2.5) shows a null-to-negative effect. We further document an ECE paradox — models discriminate better after alignment yet are less well-calibrated — and a pretraining confound, demonstrated by OLMo-2 without alignment achieving AUC within range of aligned models. The correct characterization is not "alignment improves self-knowledge" but rather "alignment improves discrimination while potentially distorting calibration, and pretraining data composition is an equally or more powerful driver."

A methodological contribution of this work is demonstrating that answer scoring procedure critically affects results: evaluating against only the canonical TriviaQA answer (rather than all gold aliases) produces a 47% false-negative rate confirmed by 100-item human audit, inflating apparent model errors and distorting the correctness labels on which P(True) is evaluated.

---

## 2. Background

### 2.1 P(True) Protocol

Given a question $q$ and a model-generated answer $a$, the model is asked to judge whether its own answer is correct. Our literal prompt is:

> {question}
>
> Is the answer "{answer}" correct? Answer with Yes or No.
>
> Answer:

P(True) is the softmax-normalized probability of the " Yes" token relative to " No" at the final prompt position; this scalar $s \in [0, 1]$ is the confidence score. Answers are the model's own greedy-decoded outputs (§3.1), making the task one of introspective self-assessment rather than answer verification from an external source. Note that this Yes/No formulation differs from the (A) True / (B) False multiple-choice variant of Kadavath et al. (2022); we retain the name P(True) for continuity with the literature. Full protocol details are given in Appendix A.

### 2.2 Domain-Stratified AUC Evaluation

We evaluate discrimination via leave-one-domain-out (LOCO) AUC. For each domain $d$, we compute AUC on the held-out domain, then average across domains. This provides a domain-robust estimate that is less sensitive to domain imbalance than overall AUC. No model is trained at any point — LOCO here is a stratified evaluation protocol, not cross-validation. TriviaQA domains are assigned by keyword matching against the question text using the following fixed keyword lists:

- **science**: science, chemistry, physics, biology, element, atom, molecule, species, organism, scientific
- **history**: history, war, battle, century, empire, revolution, president, king, queen, dynasty, treaty
- **geography**: country, capital, city, river, mountain, continent, ocean, island, largest, longest, border
- **entertainment**: film, movie, actor, actress, television, series, music, singer, band, album, song, oscar, grammy
- **sports**: sport, football, soccer, olympic, championship, player, team, tennis, golf, cricket, rugby
- **literature**: novel, book, author, poet, poem, wrote, published, fiction, playwright, shakespeare

Domain sizes in TriviaQA validation[:2000]: entertainment (n=309), geography (248), history (203), literature (108), sports (101), science (30), other (1001). The "other" category (50.1% of data) is excluded from the LOCO average due to its heterogeneous composition. Science (n=30) is likewise excluded from **all** LOCO averages: several models have too few items in one class to compute a valid science AUC, and averaging over unequal domain sets would make LOCO values non-comparable across models. All LOCO averages in this paper are therefore computed over the same five domains (history, geography, entertainment, sports, literature) for every model.

### 2.3 Item-Level Accuracy Matching

A key methodological concern is that base and instruct models differ in accuracy. If instruct models answer more questions correctly, and P(True) scores are higher for correct answers, then higher instruct AUC may reflect accuracy differences rather than improved calibration. We control for this via item-level matching: for each question, we include it in the comparison only if both the base and instruct model gave a **correct** answer (CC pair) or both gave an **incorrect** answer (WW pair). On CC pairs, both models face the same challenge of correctly assigning high confidence; on WW pairs, both should assign low confidence. This item-matched subset eliminates accuracy as a confound. Sample sizes after matching are n=798, 1464, 792, 1304 for Llama3-8B, Gemma2-2B, Gemma2-9B, and Qwen2.5-7B respectively.

### 2.4 ECE

Expected Calibration Error (ECE) measures the difference between confidence and accuracy across probability bins. We use 10 equal-width bins. ECE = 0 indicates perfect calibration; higher ECE indicates worse calibration. A model can have high AUC (good discrimination) and high ECE (poor calibration) simultaneously — this is the central tension we document. We report ECE on both the full dataset and on the item-matched subset to rule out accuracy confounds.

### 2.5 Answer Scoring

Correct evaluation of P(True) depends critically on accurate ground-truth correctness labels. TriviaQA provides multiple gold aliases for each answer (e.g., "United States of America", "USA", "the US"). A naive evaluation against only the canonical answer produces a substantial false-negative rate for instruct models, which tend to produce verbose completions. We validated this via a 100-item human audit: under canonical-only scoring, 47 of 100 randomly sampled items were false negatives — genuine correct model answers labeled as wrong. Under our alias-based scoring, all 47 were correctly identified as correct (0/100 FN).

Our scoring procedure: a model answer is marked correct if token-level F1 ≥ 0.5 against **any** gold alias, or if the normalized answer contains any normalized gold alias as a substring (containment check). Normalization removes articles (a, an, the) and punctuation. The containment check specifically addresses verbose instruct answers. All P(True) AUC and ECE results in this paper use these labels. Researchers using simpler scoring will obtain different (distorted) results.

### 2.6 Related Work

Kadavath et al. (2022) introduced P(True) and showed instruct models exhibit better-calibrated self-assessment. Kuhn et al. (2023) proposed semantic entropy as a calibration alternative that clusters semantically equivalent answers. Burns et al. (2023) showed that unsupervised probes on hidden states can recover factual knowledge without explicit prompting. Lin et al. (2022) explored verbal uncertainty expressions as calibration signals. Tian et al. (2023) documented that RLHF degrades calibration under distribution shift — consistent with our ECE paradox. The GPT-4 Technical Report (OpenAI, 2023) notes similar calibration degradation after RLHF. Xiong et al. (2024) systematically evaluated LLM uncertainty expression. Our contribution is (1) the item-matching methodology to isolate alignment effects, (2) simultaneous AUC/ECE measurement revealing divergence, and (3) pretraining controls via OLMo-2 and Pythia.

---

## 3. Experimental Setup

### 3.1 Models and Datasets

We evaluate four base/instruct pairs:

- **Llama3-8B**: Meta-Llama-3-8B (base) and Meta-Llama-3-8B-Instruct
- **Gemma2-2B**: google/gemma-2-2b (base) and google/gemma-2-2b-it
- **Gemma2-9B**: google/gemma-2-9b (base) and google/gemma-2-9b-it
- **Qwen2.5-7B**: Qwen/Qwen2.5-7B (base) and Qwen/Qwen2.5-7B-Instruct

**TriviaQA** (Joshi et al., 2017): We use the validation split, sampling n=2000 questions per model, stratified by domain. P(True) prompts use greedy-decoded model answers.

**SciQ**: Single-domain science QA dataset, n=2000. Used for cross-dataset replication. Because SciQ has a single domain, LOCO is not applicable; we report overall AUC.

Chat template ablation: instruct models additionally evaluated with their native chat template applied to the P(True) prompt.

**Generation details**: answers are generated with greedy decoding (do_sample=False, max_new_tokens=20, pad_token_id=eos_token_id). P(True) is computed as the softmax-normalized log-probability of the " Yes" token relative to " No" at the final position of the prompt. All experiments use seed=42 for bootstrap resampling and item matching.

### 3.2 Pretraining Data Control Models

To disentangle alignment effects from pretraining effects, we evaluate two base-only models:

- **OLMo-2-7B-1124** (Groeneveld et al., 2024): Trained on Dolma2 with a QA-rich mid-training phase (Dolmino; Soldaini et al., 2024). Approximately 3T tokens total. No RLHF or instruction tuning applied.
- **Pythia-6.9B** (Biderman et al., 2023): Trained on The Pile, approximately 300B tokens. No instruction tuning. The Pile has substantially less QA-focused content than Dolma2.

**Confound acknowledgment**: OLMo-2 and Pythia differ in scale (7B vs 6.9B — comparable), architecture, and pretraining data volume and composition. We cannot cleanly attribute OLMo-2's higher LOCO to data composition alone vs. architectural choices or training scale. However, both are evaluated as base models without alignment, providing the most direct available test: does a model without any RLHF/DPO achieve high P(True) AUC? The answer is yes, for OLMo-2. This is the relevant comparison for the pretraining hypothesis.

---

## 4. Results

### 4.1 TriviaQA Accuracy

| Model | Base Acc | Instruct Acc |
|---|---|---|
| Llama3-8B | 0.737 | 0.713 |
| Gemma2-2B | 0.573 | 0.551 |
| Gemma2-9B | 0.758 | 0.731 |
| Qwen2.5-7B | 0.619 | 0.603 |

Accuracy is slightly lower for instruct models in all pairs — instruct models are more verbose, which marginally increases partial-match failures under F1 scoring. All differences are small. This pattern motivates item-level matching (§2.3): the instruct AUC improvements we find are not artifacts of higher instruct accuracy.

### 4.2 ECE: Alignment Improves Llama, Worsens Others

| Model | Base ECE | Instruct ECE | Full-set direction | Matched ΔECE | Matched direction |
|---|---|---|---|---|---|
| Llama3-8B | 0.155 | 0.102 | instruct better ↓ | +0.094 | instruct worse ↑ |
| Gemma2-2B | 0.091 | 0.222 | instruct worse ↑ | +0.198 | instruct worse ↑ |
| Gemma2-9B | 0.192 | 0.226 | instruct worse ↑ | +0.322 | instruct worse ↑ |
| Qwen2.5-7B | 0.200 | 0.244 | instruct worse ↑ | −0.015 | instruct better ↓ |

ECE (10-bin, lower is better) worsens after instruction tuning in 3 of 4 families under **both** analyses, though the identity of the exception differs. On the full dataset, Llama is the sole family that improves (0.155 → 0.102). On the item-matched CC/WW subset (§2.3), which controls for accuracy differences — positive Matched ΔECE indicates higher (worse) instruct ECE on identical questions — Llama degrades (+0.094) and Qwen becomes the sole near-exception (−0.015, approximate parity). Both Gemma families degrade under both analyses (+0.198 and +0.322 matched). On SciQ (§4.6), ECE worsens in **all four** families. The robust conclusion is that calibration degradation is the dominant pattern regardless of analysis choice: this is the **ECE paradox** — instruction tuning improves rank-order discrimination (AUC, §4.4) while worsening probability calibration.

### 4.3 Full Domain-Stratified AUC (TriviaQA)

| Model | Hist. | Geo. | Ent. | Spt. | Lit. | LOCO Avg |
|---|---|---|---|---|---|---|
| Llama3-8B-base | 0.788 | 0.803 | 0.796 | 0.760 | 0.789 | 0.787 |
| Llama3-8B-instruct | 0.820 | 0.830 | 0.841 | 0.790 | 0.852 | 0.826 |
| Gemma2-2B-base | 0.536 | 0.527 | 0.463 | 0.597 | 0.564 | 0.537 |
| Gemma2-2B-instruct | 0.784 | 0.824 | 0.728 | 0.705 | 0.816 | 0.771 |
| Gemma2-9B-base | 0.755 | 0.803 | 0.740 | 0.629 | 0.840 | 0.753 |
| Gemma2-9B-instruct | 0.864 | 0.807 | 0.833 | 0.784 | 0.848 | 0.827 |
| Qwen2.5-7B-base | 0.848 | 0.786 | 0.809 | 0.823 | 0.867 | 0.827 |
| Qwen2.5-7B-instruct | 0.768 | 0.783 | 0.787 | 0.817 | 0.879 | 0.807 |
| OLMo-2-7B-base | 0.795 | 0.776 | 0.804 | 0.688 | 0.800 | 0.773 |
| Pythia-6.9B-base | 0.394 | 0.505 | 0.550 | 0.595 | 0.413 | 0.491 |

*LOCO Avg is the macro-average over the same five domains for every model (§2.2); science (n=30) is excluded from all LOCO computations due to insufficient per-class counts.*

Instruct > base in point estimate in 3 of 4 families (Llama +0.039, Gemma-2B +0.234, Gemma-9B +0.074). Qwen reverses slightly (−0.020: base 0.827 vs. instruct 0.807). OLMo-2 without alignment (0.773) exceeds two of the four base models, sits within 0.015 of Llama3-8B-base (0.787), and is comparable to Gemma2-2B-instruct (0.771); Pythia (0.491) is at chance. Notably, the three highest LOCO values in the suite — Qwen2.5-7B-base (0.827), Gemma2-9B-instruct (0.827), and Llama3-8B-instruct (0.826) — include an unaligned base model.

### 4.4 Item-Matched AUC with Bootstrap CIs (Main Results)

| Model Family | n | AUC Base | AUC Inst | Delta | 98.75% CI | Sig. |
|---|---|---|---|---|---|---|
| Llama3-8B | 798 | 0.790 | 0.856 | +0.066 | [−0.008, +0.137] | NO |
| Gemma2-2B | 1464 | 0.551 | 0.785 | +0.234 | [+0.160, +0.304] | YES |
| Gemma2-9B | 792 | 0.762 | 0.878 | +0.115 | [+0.046, +0.182] | YES |
| Qwen2.5-7B | 1304 | 0.849 | 0.819 | −0.030 | [−0.068, +0.013] | NO |

CIs at 98.75% level (Bonferroni correction: α' = 0.05/4 = 0.0125 for 4 simultaneous TriviaQA tests; two-sided CI level = 1 − α' = 98.75%). SciQ is treated as a separate replication dataset and is not included in the Bonferroni correction. Bootstrap: 1000 resamples with replacement (seed=42). Deltas and CIs are computed on unrounded values; rounded columns may differ by ±0.001.

Two of four families show statistically significant AUC improvement surviving Bonferroni correction (Gemma2-2B +0.234; Gemma2-9B +0.115). Llama3-8B is positive but does not survive correction (+0.066, CI [−0.008, +0.137]); we characterize this effect as suggestive but unconfirmed. Qwen2.5-7B shows a null-to-negative effect (−0.030, CI [−0.068, +0.013]). The Gemma2-2B delta remains the largest — consistent with the base model's near-chance discrimination (0.551) leaving substantial room for improvement.

### 4.5 Chat Template Ablation

| Model | Base LOCO | Instruct LOCO (raw) | Instruct LOCO (chat) | Chat − Raw | Chat vs. Base |
|---|---|---|---|---|---|
| Llama3-8B | 0.787 | 0.826 | 0.749 | −0.078 | base > chat-instruct |
| Gemma2-2B | 0.537 | 0.771 | 0.759 | −0.012 | chat-instruct > base |
| Gemma2-9B | 0.753 | 0.827 | 0.793 | −0.034 | chat-instruct > base |
| Qwen2.5-7B | 0.827 | 0.807 | 0.747 | −0.060 | base > chat-instruct |

Chat templates reduce LOCO AUC in all four instruct models relative to the raw prompt (−0.012 to −0.078). The instruct-vs-base comparison is **format-dependent**: under chat templates, Llama-instruct (0.749) falls below Llama-base (0.787), inverting its raw-prompt advantage; for Qwen, the base advantage already present under raw prompts (−0.020) widens under chat (0.747 vs. 0.827). Both Gemma families remain instruct-better under either format. This pattern has two implications: first, the raw prompt used in §4.4 is not conservative for instruct models — it is in fact the format under which instruct models perform best; second, the instruct AUC benefit is robust only for the Gemma families, while for Llama the direction of the effect depends on the prompt format chosen and for Qwen the base model is favored under both formats.

### 4.6 SciQ Cross-Dataset Replication

| Model | Base AUC | Instruct AUC | ΔAUC | Base ECE | Instruct ECE |
|---|---|---|---|---|---|
| Llama3-8B | 0.671 | 0.735 | +0.064 | 0.060 | 0.201 |
| Gemma2-2B | 0.549 | 0.654 | +0.105 | 0.049 | 0.300 |
| Gemma2-9B | 0.683 | 0.648 | −0.035 | 0.099 | 0.278 |
| Qwen2.5-7B | 0.715 | 0.728 | +0.013 | 0.221 | 0.272 |

*SciQ is single-domain; overall AUC reported. CIs not computed — treated as qualitative replication.*

SciQ partially replicates the TriviaQA discrimination pattern: Llama and Gemma2-2B show positive deltas; Qwen shows a near-zero delta (+0.013), consistent with the TriviaQA null; the exception is Gemma2-9B, which **inverts** on SciQ (base better by 0.035). This heterogeneity suggests domain-specific pretraining content interacts with alignment effects in ways that vary by model family. Calibration, in contrast, replicates more strongly than discrimination: **ECE worsens in all four families on SciQ** (most dramatically Gemma2-2B, 0.049 → 0.300), reinforcing the TriviaQA finding that instruction tuning degrades P(True) calibration.

### 4.7 Pretraining Data Control

| Model | Alignment | Pretraining | Accuracy | LOCO AUC |
|---|---|---|---|---|
| OLMo-2-7B-1124 | None (base only) | Dolma2 + QA-rich Dolmino mid-train (~3T tok) | 0.708 | 0.773 |
| Pythia-6.9B | None (base only) | The Pile (~300B tok) | 0.374 | 0.491 |
| Llama3-8B-instruct | RLHF/DPO | — | 0.713 | 0.826 |
| Gemma2-9B-instruct | SFT/RLHF | — | 0.731 | 0.827 |
| Qwen2.5-7B-base | None (base) | Qwen2.5 QA-rich pretraining | 0.619 | 0.827 |

OLMo-2 without any alignment achieves LOCO AUC of 0.773 — above two of the four primary base models, within 0.015 of Llama3-8B-base (0.787), and comparable to Gemma2-2B-instruct (0.771) — though below the best aligned models (0.826–0.827). Pythia without alignment scores 0.491, at chance. The 0.281 gap between two base-only unaligned models differing primarily in pretraining data composition provides direct evidence that pretraining data is a sufficient driver of P(True) discrimination. Qwen2.5-7B-base (0.827) — the highest LOCO in the entire suite, tied with Gemma2-9B-instruct — reinforces the point: an unaligned model tops the ranking. Notably, OLMo-2's full-set ECE (0.289) is the worst among the base models we evaluate: QA-rich pretraining confers discrimination, not calibration, mirroring the AUC/ECE dissociation of §5.1.

---

## 5. Discussion

### 5.1 The AUC-ECE Paradox

Instruction tuning improves P(True) discrimination (AUC) but worsens calibration (ECE) in 3/4 families. These two properties are mathematically independent: AUC measures rank-ordering ability (does P(True) assign higher scores to correct answers than incorrect ones?), while ECE measures the accuracy of probability magnitudes (does P(True) = 0.7 correspond to 70% accuracy?).

The most plausible mechanism is reward hacking on the Yes/No token distribution. RLHF and DPO training optimizes for human-preferred responses to Yes/No questions. Human raters may prefer confident, affirmative responses, which pushes the P(True) distribution toward higher values for model-generated answers. This shift improves rank-ordering (instruct answers cluster near 1, incorrect answers near 0 — better discrimination) while inflating probabilities beyond their true likelihood (worse ECE). This is consistent with Tian et al. (2023) and the GPT-4 Technical Report (OpenAI, 2023), both of which document RLHF-induced calibration degradation.

Llama's full-set exception (ECE improves: 0.155 → 0.102) does not survive item matching (matched ΔECE = +0.094), suggesting the apparent improvement partly reflects the composition of questions each model answers correctly rather than genuine calibration gains. The matched ΔECE for Qwen is slightly negative (−0.015), suggesting near-parity; the large positive matched ΔECE for Gemma-9B (+0.322) is the most striking calibration degradation case.

### 5.2 Pretraining as a Sufficient Driver

The OLMo-2 result is central evidence. A model with no alignment — no RLHF, no DPO, no instruction tuning — achieves LOCO AUC of 0.773, within 0.015 of Llama3-8B-base and inside the range spanned by aligned models (0.771–0.827). Pythia, with far less pretraining data on a less QA-rich corpus, achieves 0.491 — chance level. The primary difference between these two models is pretraining data composition and volume; we acknowledge, however, that architectural choices, training scale, and other factors may also contribute. The OLMo-2/Pythia comparison is therefore best interpreted as a demonstration that pretraining alone can produce strong P(True) discrimination — not as a controlled experiment isolating data composition as the unique cause.

Combined with Qwen2.5's null-to-negative effect — its base model attains 0.827, the highest LOCO in the suite — the pattern is consistent with a **pretraining saturation hypothesis**: models pretrained on QA-rich corpora develop strong P(True) discrimination during pretraining. For these models (Qwen2.5-7B, OLMo-2), alignment provides no additional benefit (for Qwen, a slight degradation). For models pretrained on less QA-rich data, the base model leaves room for improvement and alignment provides a boost — statistically confirmed for both Gemma families, suggestive but unconfirmed for Llama.

This reframes the Kadavath et al. finding: alignment improves P(True) not because alignment directly teaches self-knowledge, but because alignment implicitly recapitulates QA-style supervision for models whose pretraining was insufficient. Pretraining data composition may be the more fundamental driver.

### 5.3 Heterogeneity and the Qwen Exception

The heterogeneous pattern requires explanation of the Qwen null. Qwen2.5's base model is the highest-performing in our suite at LOCO 0.827; its item-matched base AUC (0.849) is likewise the highest among base models. There is simply less room for improvement. The null-to-negative effect (−0.030, CI [−0.068, +0.013]) is consistent with a ceiling effect rather than with instruction tuning having no capacity to improve P(True) in general.

The SciQ Gemma-9B inversion (base 0.683 > instruct 0.648) adds complexity. It suggests that the alignment benefit may be domain-specific: Gemma-9B's alignment may have been more effective on trivia-style QA (TriviaQA) than on structured science questions (SciQ), or SciQ's domain may interact differently with the model's pretraining distribution. This heterogeneity across datasets and models is itself an important finding — P(True) quality is not a fixed property of a model but varies with domain and dataset.

### 5.4 Prompt Format: Format-Dependent Elicitation

Chat templates reduce LOCO AUC in all four instruct models relative to the raw prompt. The commonly assumed direction — that instruct models benefit from familiar chat format — is not supported. The instruct advantage from §4.4 was obtained with raw prompts, which are less familiar to instruct models.

More importantly, the chat-template comparison reveals format-dependence in the direction of the base/instruct gap: under raw prompts, instruct > base for 3/4 families (§4.3). Under chat templates, Llama-instruct (0.749) falls below Llama-base (0.787), inverting its raw-prompt advantage; Qwen-base is favored under both formats (raw −0.020, chat 0.747 vs. 0.827). The two Gemma families maintain the instruct advantage under both formats.

This implies the instruct AUC benefit is **robust for Gemma models, fragile for Llama (positive under raw prompts, reversed under chat templates), and absent-to-reversed for Qwen under both formats**. Researchers using P(True) with instruct models should test multiple prompt formats; format choice can change not just the magnitude but the direction of the base/instruct comparison.

### 5.5 Implications for P(True) Research

1. **Use all gold aliases**: A canonical-only evaluation of TriviaQA produces a 47% false-negative rate (100-item audit). Any P(True) study using TriviaQA should use F1 ≥ 0.5 against all aliases plus containment check.

2. **Report ECE alongside AUC**: AUC and ECE measure different things and can diverge. A paper reporting only AUC would miss the calibration degradation; a paper reporting only ECE would miss the discrimination improvement.

3. **Apply item-level matching**: Without matching, accuracy differences between base and instruct models confound AUC comparisons. Item-matched analysis is essential for causal claims about alignment effects.

4. **Use Bonferroni correction**: With 4 model families tested, uncorrected p-values overstate evidence. Under α' = 0.0125 (98.75% CI), two of our four family effects survive; a third (Llama) is positive but unconfirmed.

5. **Include pretraining controls**: OLMo-2 and Pythia demonstrate that aligned/unaligned comparison within families is not the only relevant test. Base-only models with different pretraining provide a direct test of pretraining effects.

6. **Test multiple prompt formats**: As §5.4 shows, base/instruct direction can reverse under different formats. Format ablation should be standard practice.

### 5.6 Limitations

1. **Science domain n=30**: TriviaQA science domain contains only 30 items. AUC estimates from this domain have low reliability; science-domain results should not be interpreted in isolation.

2. **OLMo-2 vs. Pythia confounds**: These models differ in scale, architecture, and data volume in addition to data composition. The comparison is a proxy test, not a controlled experiment.

3. **SciQ CIs not computed**: SciQ results are qualitative replications. Item-matched CIs for SciQ were not computed; significance claims for SciQ are not made.

4. **Chat template item matching**: The chat template ablation uses the same correctness labels as the raw-prompt experiment. Ideally, item matching for chat-template results would use answers generated under chat template format.

5. **F1+containment scope**: Our 100-item human audit found 0 false negatives under our scoring scheme; the 47 items flagged as incorrect under canonical-only scoring were confirmed by human inspection to be genuinely **correct** answers (i.e., false negatives of canonical scoring). This validates the higher FN rate of canonical scoring but is not a direct validation of our scoring's false-positive rate, which remains unaudited.

---

## 6. Conclusion

We have conducted a controlled study of instruction tuning effects on P(True) self-assessment quality. Our findings:

1. **Instruction tuning significantly improves P(True) AUC in 2 of 4 model families** (Gemma2-2B: +0.234; Gemma2-9B: +0.115; both surviving Bonferroni correction at the 98.75% CI level), with a positive but non-significant trend for Llama3-8B (+0.066) and a null-to-negative effect for Qwen2.5-7B (−0.030). This partially confirms Kadavath et al. (2022) under stronger methodological controls: the direction replicates in 3 of 4 families, but the effect is neither universal nor uniform.

2. **ECE worsens in 3 of 4 families on TriviaQA and in all 4 on SciQ** after instruction tuning. AUC and ECE diverge — models discriminate better while becoming less calibrated. Researchers equating AUC with "self-knowledge" should be aware of this distinction.

3. **Qwen2.5-7B shows a null-to-negative effect** (−0.030, CI [−0.068, +0.013]), consistent with pretraining saturation: its base model attains the highest LOCO AUC in the suite (0.827).

4. **OLMo-2 without alignment achieves LOCO AUC of 0.773** — within 0.015 of Llama3-8B-base and inside the range of aligned models — while Pythia achieves only 0.491, at chance. The 0.281 gap between two unaligned models suggests pretraining data composition is a sufficient driver of P(True) discrimination.

5. **Prompt format is a confound**: chat templates reduce LOCO AUC for all four instruct models; the base/instruct direction reverses for Llama under chat templates, Qwen favors base under both formats, and both Gemma families remain instruct-better under either format.

These results suggest that what is often attributed to alignment as self-knowledge is better characterized as an interaction between pretraining data composition and alignment fine-tuning. Pretraining on QA-rich corpora may be a more reliable path to high-quality P(True) discrimination than alignment alone — while alignment tends to degrade calibration.

---

## Reproducibility Statement

Code, literal prompt templates, run configurations, and the consolidated results file (`results.json`) from which every table and every in-text numeric value in this paper is generated will be released at **https://github.com/rodrigoignaci0/neos-research**. All tables in this document are generated programmatically from `results.json`; no numeric value is hand-edited.

---

## Appendix A: P(True) Prompt and Protocol Details

**P(True) prompt (raw condition, all models):**

```
{question}

Is the answer "{answer}" correct? Answer with Yes or No.

Answer:
```

**Scored tokens:** softmax-normalized probability of " Yes" relative to " No" at the final prompt position.

**Chat-template condition:** the same prompt delivered through each instruct model's native chat template, with the same token pair scored at the assistant response position.

**Answer generation:** greedy decoding (`do_sample=False`, `max_new_tokens=20`, `pad_token_id=eos_token_id`).

**Correctness labels:** an answer is correct if token-level F1 ≥ 0.5 against any gold alias, or if the normalized answer contains any normalized gold alias as a substring (articles and punctuation removed).

**LOCO domains:** history, geography, entertainment, sports, literature; science (n=30) and "other" excluded (§2.2).

**Statistics:** bootstrap with 1000 resamples (seed=42); Bonferroni α' = 0.05/4 = 0.0125 → 98.75% CIs; SciQ treated as replication, outside the correction family.

---

## References

Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., Khan, K., Purohit, R., Prashanth, U., Raff, E., Skowron, A., Sutawika, L., and van der Wal, O. (2023). Pythia: A suite for analyzing large language models across training and scaling. *ICML 2023*.

Burns, C., Ye, H., Klein, D., and Steinhardt, J. (2023). Discovering latent knowledge in language models without supervision. *ICLR 2023*.

Campos Vargas, R. (2026). Hidden state probes learn domain identity, not uncertainty. *EMNLP 2026 Workshop BlackboxNLP*.

Groeneveld, D., Magnusson, I., Walsh, A., Bhagia, A., Schwartz, R., Tafjord, O., Jha, P., Liu, H., Weissenberger, D., Thomas, C., et al. (2024). OLMo: Accelerating the science of language models. *ACL 2024*.

Joshi, M., Choi, E., Weld, D. S., and Zettlemoyer, L. (2017). TriviaQA: A reading comprehension dataset over trivia questions. *ACL 2017*.

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Tran-Johnson, E., Johnston, S., El-Showk, S., Jones, A., Elhage, N., Hume, T., Chen, A., Bai, Y., Bowman, S., Fort, S., Ganguli, D., Hernandez, D., Jacobson, J., Kernion, J., Lovitt, L., Ndousse, K., Olsson, C., Ringer, S., Amodei, D., Amodei, D., Clark, J., McCandlish, S., Olah, C., and Kaplan, J. (2022). Language models (mostly) know what they know. *arXiv:2207.05221*.

Kuhn, L., Gal, Y., and Farquhar, S. (2023). Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. *ICLR 2023*.

Lin, S., Hilton, J., and Evans, O. (2022). Teaching models to express their uncertainty in words. *TMLR 2022*.

OpenAI. (2023). GPT-4 Technical Report. *arXiv:2303.08774*.

Qwen Team. (2024). Qwen2.5 technical report. *arXiv:2412.15218*.

Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. (2016). SQuAD: 100,000+ questions for machine comprehension of text. *EMNLP 2016*.

Soldaini, L., Kinney, R., Bhagia, A., Schwartz, R., Atkinson, D., Authur, R., Chandu, K. R., Doğan, B., Fries, J., Fuentes, C., et al. (2024). Dolma: An open corpus of three trillion tokens for language model pretraining research. *arXiv:2402.00159*.

Tian, K., Mitchell, E., Yao, H., Manning, C. D., and Finn, C. (2023). Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. *EMNLP 2023*.

Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., and Hooi, B. (2024). Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. *ICLR 2024*.
