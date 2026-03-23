# lobe.jl — Lightweight lobe state for NERO relevance scoring
#
# Decouples SpikenautNero from the full EnsembleBrain / CUDA stack.
# Populate from whatever SNN backend you use, then pass to update_relevance!.

"""
    LobeState

Minimal per-lobe summary consumed by NERO each tick.

Fields:
  last_spike_rate  — normalised firing rate [0,1] for this lobe this tick
  output           — N_OUT-element readout vector (CPU Float32)
"""
struct LobeState
    last_spike_rate::Float32
    output::Vector{Float32}
end

"""
    LobeState(n_out::Int) -> LobeState

Construct a zero-initialised LobeState with `n_out` output channels.
"""
LobeState(n_out::Int) = LobeState(0.0f0, zeros(Float32, n_out))
