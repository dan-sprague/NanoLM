module NanoLM

using LinearAlgebra
using Random
using Distributions

"""
Abstract type for energy-based language models
Allows different energy functions to be implemented
"""
abstract type EnergyBasedLM end

"""
Autoregressive energy-based LM with general energy function
Parameters:
- V: vocabulary (set of possible token values)
- n: sequence length
- B: temperature parameter
- energy_fn: function that computes E(x_k | x_{<k})
"""
struct AutoregressiveEBLM{T} <: EnergyBasedLM
    V::Vector{Int64}                    # Vocabulary
    n::Int                         # Sequence length
    B::Float64                     # Temperature parameter
    π::Vector{Float64}              # Prior distribution over vocabulary
    energy_fn::Function            # E(x_k | x_{<k})
end

"""
Example: General sequence model with arbitrary pairwise interactions
E(x_k | x_{<k}) = -∑_{j<k} J[j,k] * x_k * x_j
"""
function AutoregressiveEBLM(n::Int, B::Float64, J::Matrix{Float64}, V::Vector,π::Vector)
    
    function energy(x_k, prefix, k)
        -sum(J[j,k] * x_k * prefix[j] for j in 1:k-1)
    end
    
    return AutoregressiveEBLM(V, n, B, π,energy)
end

include("math.jl")



end # module NanoLM
