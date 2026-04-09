# nero_orchestrator.jl — NERO: Neuromorphic Evaluation of Relevance and Orchestration
#
# NERO manages a 4-node static graph over lobe states and computes a per-tick
# relevance score for each lobe based on two signals:
#
#   1. Spike Density  — normalised firing rate of the lobe (0..1).
#      A lobe that is firing actively is "engaged" with the current regime.
#
#   2. Manifold Surprise — how much the lobe's current readout deviates from its
#      recent exponential moving average (EMA).  High surprise → the lobe just
#      transitioned into a new attractor / regime.
#
# Final relevance = α × spike_density + β × manifold_surprise + γ × ema_momentum
# All scalars are Float32; no heap allocation in the hot path.
#
# Wire contract (appended to the existing 72-byte readout):
#   [72..76]  relevance[1]  (Float32 LE)
#   [76..80]  relevance[2]  (Float32 LE)
#   [80..84]  relevance[3]  (Float32 LE)
#   [84..88]  relevance[4]  (Float32 LE)
#
# ─────────────────────────────────────────────────────────────────────────────

using LinearAlgebra: norm
using Printf

# ── NERO Tuning Constants ─────────────────────────────────────────────────────

const NERO_ALPHA       = 0.50f0   # Weight: spike density contribution
const NERO_BETA        = 0.35f0   # Weight: manifold surprise contribution
const NERO_GAMMA       = 0.15f0   # Weight: readout EMA momentum
const NERO_EMA_DECAY   = 0.05f0   # EMA smoothing factor (5% new estimate per tick)
const NERO_MIN_SCORE   = 0.01f0   # Clamp: never let any lobe drop to zero relevance
const NERO_EPSILON     = 1.0f-6   # Numerical stability floor for normalisation

# Default lobe names (4-lobe ensemble for LLM routing)
const NERO_DEFAULT_LOBE_NAMES = ["Attention", "FFN", "Memory", "Output"]

# Cross-lobe inhibition: lateral inhibition for winner-take-all routing.
# Layout: NERO_INHIBIT[from_lobe, to_lobe]. Higher values = stronger suppression.
const NERO_INHIBIT = Float32[
    0.0   0.08  0.05  0.02;   # Attention → {FFN, Memory, Output}
    0.04  0.0   0.06  0.03;   # FFN      → {Attention, Memory, Output}
    0.02  0.03  0.0   0.05;   # Memory   → {Attention, FFN, Output}
    0.01  0.02  0.03  0.0     # Output   → {Attention, FFN, Memory}
]

# ── NERO State ────────────────────────────────────────────────────────────────

"""
    NeroOrchestrator

Holds all mutable state for NERO's per-tick relevance computation.
Pre-allocated at startup; the hot-path `update_relevance!` does NO heap
allocation — all work is in-place on these fields.

Fields:
  n_lobes          — number of lobes (default 4)
  n_out            — readout width per lobe (default 16)
  lobe_names       — human-readable lobe labels
  adjacency_matrix — n_lobes × n_lobes directed adjacency weights
  routing_weights  — current routing weights vector (sums to 1.0)
  readout_ema      — per-lobe EMA of the readout (n_lobes × n_out)
  spike_density    — current spike density per lobe
  prev_routing_weights — previous tick routing weights (for momentum)
  surprise         — manifold surprise score per lobe
  scratch          — reusable scratch buffer (n_out elements)
  tick_count       — global tick counter
"""
mutable struct NeroOrchestrator
    n_lobes::Int
    n_out::Int
    lobe_names::Vector{String}
    adjacency_matrix::Matrix{Float32}
    routing_weights::Vector{Float32}
    readout_ema::Matrix{Float32}
    spike_density::Vector{Float32}
    prev_routing_weights::Vector{Float32}
    surprise::Vector{Float32}
    scratch::Vector{Float32}
    tick_count::Int64
end

"""
    NeroOrchestrator(; n_lobes=4, n_out=16, lobe_names=NERO_DEFAULT_LOBE_NAMES) -> NeroOrchestrator

Build the static lobe graph and pre-allocate all working buffers.
"""
function NeroOrchestrator(;
        n_lobes::Int = 4,
        n_out::Int   = 16,
        lobe_names::Vector{String} = NERO_DEFAULT_LOBE_NAMES)

    adjacency_matrix = zeros(Float32, n_lobes, n_lobes)
    for i in 1:n_lobes, j in 1:n_lobes
        i != j && (adjacency_matrix[i, j] = 1.0f0)
    end

    NeroOrchestrator(
        n_lobes,
        n_out,
        lobe_names,
        adjacency_matrix,
        fill(1.0f0 / n_lobes, n_lobes),    # equal routing weights at start
        zeros(Float32, n_lobes, n_out),
        zeros(Float32, n_lobes),
        fill(1.0f0 / n_lobes, n_lobes),
        zeros(Float32, n_lobes),
        zeros(Float32, n_out),
        Int64(0)
    )
end

# ── Core Update ───────────────────────────────────────────────────────────────

"""
    update_relevance!(nero, lobes) -> nothing

Compute NERO relevance scores for all lobes from the current lobe states.
`lobes` is a `Vector{LobeState}` — one per lobe.

The result is stored in `nero.routing_weights` (n_lobes × Float32).

Algorithm per lobe i:
  1. spike_density[i]  = lobes[i].last_spike_rate
  2. readout_ema[i,:]  = (1-EMA_DECAY)×old_ema + EMA_DECAY×lobes[i].output
  3. surprise[i]       = norm(readout_delta) / (norm(readout_ema) + ε)
  4. momentum[i]       = |routing_weights[i] - prev_routing_weights[i]|
  5. raw[i]            = α×density + β×surprise + γ×momentum

Cross-lobe inhibition:
  6. inhibited[i] = raw[i] - Σⱼ NERO_INHIBIT[j,i] × raw[j]

Softmax normalisation → sum(relevance) = 1.0, each ≥ NERO_MIN_SCORE.
"""
function update_relevance!(nero::NeroOrchestrator, lobes::Vector{LobeState})
    nero.tick_count += 1
    n = nero.n_lobes
    raw = nero.prev_relevance   # reuse buffer (prev no longer needed this tick)

    # ── Stage 1-3: per-lobe signal collection ─────────────────────────────
    for i in 1:n
        lobe = lobes[i]

        # 1. Spike density
        spike_density = lobe.last_spike_rate

        # 2. Readout EMA update (in-place)
        copyto!(nero.scratch, lobe.output)
        @views ema_row = nero.readout_ema[i, :]
        ema_row .= (1.0f0 - NERO_EMA_DECAY) .* ema_row .+
                    NERO_EMA_DECAY .* nero.scratch

        # 3. Manifold surprise: |new - ema| / (|ema| + ε)
        nero.scratch .-= ema_row      # scratch ← delta
        delta_norm = norm(nero.scratch)
        ema_norm   = norm(ema_row) + NERO_EPSILON
        nero.surprise[i] = delta_norm / ema_norm

        # 4. Momentum
        momentum = abs(nero.routing_weights[i] - nero.prev_routing_weights[i])

        # 5. Raw score
        raw[i] = NERO_ALPHA * spike_density +
                 NERO_BETA  * nero.surprise[i] +
                 NERO_GAMMA * momentum
    end

    # ── Stage 4: cross-lobe graph inhibition ──────────────────────────────
    inhibited = nero.routing_weights
    for dst in 1:n
        inh_sum = 0.0f0
        for src in 1:n
            if nero.adjacency_matrix[src, dst] > 0.0f0
                inh_sum += NERO_INHIBIT[src, dst] * raw[src]
            end
        end
        inhibited[dst] = max(raw[dst] - inh_sum, NERO_MIN_SCORE)
    end

    # ── Stage 5: softmax normalisation ────────────────────────────────────
    max_val = maximum(inhibited)
    s = 0.0f0
    for i in 1:n
        inhibited[i] = exp(inhibited[i] - max_val)
        s += inhibited[i]
    end
    inhibited ./= (s + NERO_EPSILON)

    for i in 1:n
        if inhibited[i] < NERO_MIN_SCORE
            inhibited[i] = NERO_MIN_SCORE
        end
    end
    inhibited ./= (sum(inhibited) + NERO_EPSILON)

    copyto!(nero.prev_routing_weights, nero.routing_weights)
    return nothing
end

# ── Diagnostics ───────────────────────────────────────────────────────────────

"""
    nero_diagnostics(nero::NeroOrchestrator) -> String

One-line NERO state summary for logging.
"""
function nero_diagnostics(nero::NeroOrchestrator)::String
    lobe_strs = [@sprintf("%s=%.2f", nero.lobe_names[i], nero.routing_weights[i])
                 for i in 1:nero.n_lobes]
    dominant = argmax(nero.routing_weights)
    surprise_str = join([@sprintf("%.3f", nero.surprise[i]) for i in 1:nero.n_lobes], ",")
    @sprintf("[NERO tick=%d] %s | dominant=%s | surprise=[%s]",
        nero.tick_count,
        join(lobe_strs, " "),
        nero.lobe_names[dominant],
        surprise_str)
end
