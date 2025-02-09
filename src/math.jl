"""
Compute conditional probability distribution
p(x_k | x_{<k}) = exp(-B * E(x_k | x_{<k})) / Z_k
where Z_k = ∑_{v ∈ V} exp(-B * E(v | x_{<k}))
"""
function conditional_distribution(model::AutoregressiveEBLM, prefix::Vector)
    k = length(prefix)
    
    # Compute unnormalized probabilities for each vocabulary item
    energies = [model.energy_fn(voc_item, prefix, k) for voc_item in model.V]
    unnormalized = exp.(-model.B * energies)
    
    # Normalize
    Z = sum(unnormalized)
    return unnormalized ./ Z
end

"""
Generate sequence by sampling token by token:
x_k ~ p(· | x_{<k})
"""
function generate(model::AutoregressiveEBLM)
    sequence = zeros(Int64,model.n)
    sequence[1] = model.V[rand(Categorical(model.π))]

    for k in 2:model.n
        # Get distribution over next token
        probs = conditional_distribution(model, sequence)
        
        # Sample and append token
        idx = rand(Categorical(probs))
        sequence[k] = model.V[idx]
    end
    
    return sequence
end

"""
Compute probability of a complete sequence:
p(x) = ∏_{k=1}^n p(x_k | x_{<k})
"""
function sequence_probability(model::AutoregressiveEBLM, sequence)
    log_prob = 0.0

    log_prob += log(model.π[argmax(V .== sequence[1])])
    
    for k in 2:lastindex(sequence)
        prefix = sequence[1:k-1]
        probs = conditional_distribution(model, prefix)
        token_idx = argmax(V .== sequence[k])
        log_prob += log(probs[token_idx])
    end
    
    return exp(log_prob)
end
