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
struct AutoregressiveEBLM <: EnergyBasedLM
    V::Vector{Float64}                    # Vocabulary
    n::Int                         # Sequence length
    B::Float64                     # Temperature parameter
    π::Vector{Float64}              # Prior distribution over vocabulary
    J::Matrix{Float64}              # Interaction matrix
end


function (model::AutoregressiveEBLM)(x_k, prefix,k)
    -sum(@. model.J[k,:] * x_k * prefix)
end


struct EBLM <: EnergyBasedLM
    V::Vector{Float64}                    # Vocabulary
    n::Int                         # Sequence length
    β::Float64                     # Temperature parameter
    π::Vector{Float64}              # Prior distribution over vocabulary
    J::Matrix{Float64}

    U::Function
    ∇U::Function
end

function EBLM(V,n,β,π,J)
    U(x) = -β*(x' * J * x)
    ∇U(x) = -β*(J + J')*x

    EBLM(V,n,β,π,J,U,∇U)
end



include("math.jl")



end # module NanoLM
