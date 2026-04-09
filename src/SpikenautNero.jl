"""
    SpikenautNero

NERO: Neuromorphic Evaluation of Relevance and Orchestration.

Computes per-tick relevance scores across SNN lobes using:
- Spike density (α=0.50)
- Manifold surprise via EMA deviation (β=0.35)
- Relevance momentum (γ=0.15)

Cross-lobe inhibition via a static directed graph, followed by
softmax normalisation so all scores sum to 1.0.

## Provenance

Extracted from Eagle-Lander, the author's own private neuromorphic GPU supervisor
repository (closed-source). NERO orchestrated a 4-lobe 65,536-neuron LSM ensemble
in production before being open-sourced as a standalone Julia package.
"""
module SpikenautNero

export LobeState, NeroOrchestrator, update_relevance!, nero_diagnostics, adapt_leak!

include("lobe.jl")
include("nero_orchestrator.jl")

end # module
