<p align="center">
  <img src="docs/logo.png" width="220" alt="Spikenaut">
</p>

<h1 align="center">SpikenautNero.jl</h1>
<p align="center">Neuromorphic Attention Router and Sparsity Enforcer for LLM-SNN fusion</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Julia-9558B2" alt="Julia">
  <img src="https://img.shields.io/badge/license-GPL--3.0-orange" alt="GPL-3.0">
</p>

---

Lightweight, zero-allocation neuromorphic routing engine for LLM-SNN hybrid architectures.
NERO (Neuromorphic Evaluation of Relevance and Orchestration) dynamically routes compute
through attention layers using spike density, manifold surprise, and momentum signals.
Enables hardware-software co-design with thermal-aware sparsity control.

## Features

- `NeroOrchestrator` — N-component routing weights with configurable weights (α, β, γ)
- `update_relevance!(orch, readouts)` — online update from new lobe readouts
- `adapt_leak!(leak_rate, fan_speed_perc)` — thermal-aware sparsity control
- Cross-lobe inhibition via lateral inhibition (winner-take-all routing)
- Manifold surprise detection: `||readout - ema|| / ||ema||`
- Numerically stable softmax with floor clamping (no component goes fully silent)
- Zero-allocation hot path (pre-allocated buffers, in-place operations)
- Hardware-software co-design: responds to PCIe spikes and thermal telemetry

## Installation

```julia
using Pkg
Pkg.add("SpikenautNero")
```

## Quick Start

```julia
using SpikenautNero

# 4-lobe neuromorphic router (Attention, FFN, Memory, Output)
orch = NeroOrchestrator(
    n_lobes   = 4,
    alpha     = 0.50,  # spike density weight
    beta      = 0.35,  # manifold surprise weight  
    gamma     = 0.15   # momentum weight
)

# Update every tick from SNN readouts
update_relevance!(orch, lobe_readouts)  # lobe_readouts: Vector{LobeState}

# Get routing weights for LLM attention gating
routing_weights = orch.routing_weights  # [0,1]^4 — softmax-normalized

# Thermal adaptation (optional)
leak_rate = Ref{Float32}(0.05f0)
adapt_leak!(leak_rate, fan_speed_perc)  # 0-100 from hardware telemetry
```

## NERO Routing Formula

```
score_i = α · density_i  +  β · surprise_i  +  γ · momentum_i

density_i  = mean(spikes_i[t])
surprise_i = ||r_i - ema_i|| / ||ema_i||
momentum_i = |routing_weights[i] - prev_routing_weights[i]|

inhibited  = score_i - Σ_j C_ij · score_j
final      = softmax(inhibited, floor=0.05)
```

## Hardware-Software Co-Design

NERO enables dynamic compute throttling based on hardware conditions:
- **PCIe spikes** trigger excitatory signals to the SNN
- **Thermal telemetry** (fan speed) modulates neuron leak rates via `adapt_leak!`
- **Cross-lobe inhibition** can suppress heavy attention layers under load
- **Winner-take-all routing** routes compute through lighter RNN layers when stressed

*Dopamine (Schultz, 1998); Cortisol/inhibition (Arnsten, 2009); Acetylcholine/focus (Hasselmo, 1999)*

## Extracted from Production

Extracted from [Eagle-Lander](https://github.com/rmems/Eagle-Lander), a private neuromorphic
GPU supervisor. NERO orchestrated a 4-lobe 65,536-neuron LSM ensemble in production
before being open-sourced as a standalone Julia package for LLM-SNN fusion.

## Part of the Spikenaut Ecosystem

| Library | Purpose |
|---------|---------|
| [SpikenautLSM.jl](https://github.com/rmems/SpikenautLSM.jl) | GPU sparse reservoir (provides readouts) |
| [SpikenautDistill.jl](https://github.com/rmems/SpikenautDistill.jl) | Training + FPGA export |
| [spikenaut-backend](https://github.com/rmems/spikenaut-backend) | Rust NERO packet consumer |

## License

GPL-3.0-or-later
