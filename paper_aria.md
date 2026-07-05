# ARIA: Adaptive Regions with Internal Awareness

**Rodrigo Campos Vargas**
NE-OS SpA, Chile — July 2026

---

## Abstract

ARIA combines RT-LI's three-region architecture with recurrent feedback, distributed associative memory, and test-time weight editing into a unified 258M-parameter language model. The architecture makes explicit the three computational phases universally observed in transformer internals, adds bidirectional feedback between regions, equips the model with a distributed memory system (DAMA) of 8 specialized agents, and applies local weight updates during inference when uncertainty exceeds a learned threshold (TTT). Training uses a 3-phase curriculum with auxiliary losses: JEPA (latent prediction), contrastive consistency, and uncertainty calibration.

---

## Architecture

### Three Regions (parallel, connected by gates)

- **Region A — Chaos Encoder** (4L/8H): processes raw token embeddings
- **Region B — Semantic Integrator** (8L/16H + uncertainty head + TTT layer): organizes semantic content, estimates uncertainty per token
- **Region C — Crystal Decoder** (4L/8H): crystallizes output representations

### Inter-region Gates

- **A→B** (LateralGate): feedforward signal with learned α=0.1 scaling
- **B→C** (LateralGate): feedforward signal
- **C→B** (FeedbackGate): feedback from C to B for next cycle (recurrent)

N_CYCLES=1 (extensible to 3+)

### DAMA Memory

8 specialized agents × 64 dimensions × 512 slots. Each agent learns to encode/decode a semantic fragment. Memory queried by attention over slots, written via EMA when uncertainty > threshold. Agent weights learned end-to-end.

### TTT Layer

Fast-weight layer in Region B. During inference, when uncertainty > 0.45, computes local gradient step to update W from h→target. Scales correction by uncertainty magnitude.

### Curriculum (3 phases)

- Phase 1 (0–30K steps): Region A+B only — basic structure
- Phase 2 (30–70K steps): +Region C — crystallization
- Phase 3 (70–100K steps): +DAMA+TTT — full emergent capabilities

### Loss

```
L = L_ce + 0.10*L_crystal + 0.05*L_uncertainty + 0.02*L_dama + 0.05*L_jepa + 0.05*L_consistency
```

- **L_crystal**: penalizes if Region C does not improve over Region B
- **L_uncertainty**: calibrates uncertainty head against correctness signal
- **L_dama**: DAMA recall should approximate h_b when uncertainty is high
- **L_jepa**: h_b[:,:-k] predicts h_b[:,k:].detach() (future latent, not tokens)
- **L_consistency**: first/second half of same document should have similar representations (contrastive)

---

## Parameters

| Component | Params |
|---|---|
| Region A (4L/8H) | ~25M |
| Region B (8L/16H) | ~134M |
| Region C (4L/8H) | ~25M |
| DAMA (8 agents) | ~0.5M |
| Gates + heads | ~2M |
| Embeddings | ~52M |
| **Total** | **~258M** |

---

## Status

Architecture implemented and tested. Training paused pending RT-LI validation via noise floor characterization experiment (noise_floor_v2: 4 seeds baseline + 2 seeds RT-LI × 30K steps).

**Code:** [`aria.py`](https://github.com/rodrigoignaci0/neos-research) | [`train_aria.py`](https://github.com/rodrigoignaci0/neos-research)
