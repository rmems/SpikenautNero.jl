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

"""
    adapt_leak!(leak_rate::Ref{Float32}, fan_speed_perc::Float32) -> nothing

Adapt the base leak rate based on hardware thermal stress.

As fan speed increases (indicating system stress / heat), the leak rate increases.
Higher leak makes neurons harder to fire, naturally inducing sparsity and reducing
power consumption. This enables hardware-software co-design where the SNN
dynamically responds to thermal conditions.

Arguments:
  leak_rate      - Reference to the current leak rate (modified in-place)
  fan_speed_perc - Fan speed percentage [0..100] from hardware telemetry
"""
function adapt_leak!(leak_rate::Ref{Float32}, fan_speed_perc::Float32)
    min_leak = 0.01f0
    max_leak = 0.25f0
    normalized = clamp(fan_speed_perc / 100.0f0, 0.0f0, 1.0f0)
    leak_rate[] = min_leak + normalized * (max_leak - min_leak)
    return nothing
end
