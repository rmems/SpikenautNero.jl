<p align="center">
  <img src="docs/logo.png" width="220" alt="Spikenaut">
</p>

<h1 align="center">SpikenautNero.jl</h1>
<p align="center">Multi-lobe relevance scoring with cross-inhibition for ensemble neural systems</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Julia-9558B2" alt="Julia">
  <img src="https://img.shields.io/badge/license-GPL--3.0-orange" alt="GPL-3.0">
</p>

---

Lightweight, zero-allocation relevance scoring engine for multi-model ensemble systems.
Computes per-component importance from spike density, manifold surprise, and momentum,
with cross-component inhibition and softmax normalization. Works for any ensemble — not
just spiking networks.

## Features

- `NeroOrchestrator` — N-component relevance scoring with configurable weights (α, β, γ)
- `update_relevance!(orch, readouts)` — online update from new lobe readouts
- Manifold surprise: `||readout - ema|| / ||ema||` — works for any vector output
- Configurable cross-component inhibition matrix (sparse or dense)
- Numerically stable softmax with floor clamping (no component goes fully silent)
- Zero-allocation hot path (pre-allocated buffers, in-place operations)
- Wire-protocol helpers for IPC binary publishing

## Installation

```julia
using Pkg
Pkg.add("SpikenautNero")
```

## Quick Start

```julia
using SpikenautNero

# 4-lobe ensemble orchestrator
orch = NeroOrchestrator(
    n_lobes   = 4,
    alpha     = 0.4,   # spike density weight
    beta      = 0.3,   # manifold surprise weight
    gamma     = 0.3,   # momentum weight
    inhibition = [0 0.1 0.1 0.1;
                  0.1 0 0.1 0.1;
                  0.1 0.1 0 0.1;
                  0.1 0.1 0.1 0]
)

# Update every tick
update_relevance!(orch, lobe_readouts)  # lobe_readouts: N_neurons × N_lobes

scores = orch.scores  # [0,1]^4 — softmax-normalized relevance
```

## NERO Scoring Formula

```
score_i = α · density_i  +  β · surprise_i  +  γ · momentum_i

density_i  = mean(spikes_i[t])
surprise_i = ||r_i - ema_i|| / ||ema_i||
momentum_i = ema_alpha * momentum_i + (1 - ema_alpha) * score_i

inhibited  = score_i - Σ_j C_ij · score_j
final      = softmax(inhibited, floor=0.05)
```

*Dopamine (Schultz, 1998); Cortisol/inhibition (Arnsten, 2009); Acetylcholine/focus (Hasselmo, 1999)*

## Extracted from Production

Extracted from [Eagle-Lander](https://github.com/rmems/Eagle-Lander), a private 4-lobe
neuromorphic brain. The NERO scoring algorithm was decoupled from lobe-specific
signal semantics so it works with any ensemble of vector-valued components.

## Part of the Spikenaut Ecosystem

| Library | Purpose |
|---------|---------|
| [SpikenautLSM.jl](https://github.com/rmems/SpikenautLSM.jl) | GPU sparse reservoir (provides readouts) |
| [SpikenautDistill.jl](https://github.com/rmems/SpikenautDistill.jl) | Training + FPGA export |
| [spikenaut-backend](https://github.com/rmems/spikenaut-backend) | Rust NERO packet consumer |

## License

GPL-3.0-or-later
