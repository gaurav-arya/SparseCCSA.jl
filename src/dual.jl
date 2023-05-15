
"""
A callable structure for evaluating the dual function and its gradient.
"""
@kwdef struct DualEvaluator{T, L}
    cache::CCSACache{T, L}
end

"""
    (evaluator::DualEvaluator{T})(neg_gλ, neg_grad_gλ, λ)

Set the negated dual gradient neg_grad_gλ [shape 1 x m] and negated dual objective neg_gλ [length = 1]
in-place to their new values at λ [length = m].
The evaluator's internal buffer Δx is also set so that xᵏ + Δx is the optimal primal.
λ = nothing can be specified when the dual problem is of dimension-0.
"""
function (evaluator::DualEvaluator{T})(neg_gλ, neg_grad_gλ, λ) where {T}
    @unpack σ, ρ, x, fx, jac_fx, lb, ub = evaluator.cache # extract "read-only" info specifying dual problem
    @unpack a, b, Δx, λ_all, grad_gλ_all = evaluator.cache # extract dual buffers
    # The dual evaluation turns out to be simpler to express with λ_{1...m} left-augmented by λ_0 = 1.
    # We have special (m+1)-length buffers for this, which is a little wasteful. 
    λ_all[1] = 1
    (λ !== nothing) && (λ_all[2:end] .= λ)

    a .= dot(λ_all, ρ) ./ (2 .* σ .^ 2)
    mul!(b, jac_fx', λ_all)
    @. Δx = -b / (2 * a)
    @. Δx = clamp(Δx, -σ, σ)
    @. Δx = clamp(Δx, lb - x, ub - x)

    # For dimension-0 dual dual problem allow early break when λ = nothing.
    (λ === nothing) && return nothing

    if (neg_grad_gλ !== nothing) && (length(neg_grad_gλ) > 0)
        # Below we populate grad_gλ_all, i.e. the gradient WRT λ_0, ..., λ_m,
        # although we ultimately don't care about the first entry.
        grad_gλ_all .= 0
        mul!(grad_gλ_all, jac_fx, Δx)
        grad_gλ_all .+= fx .+ mapreduce(abs2 ∘ /, +, Δx, σ; init=zero(T)) ./ 2 .* ρ
        neg_grad_gλ .= -1 * (@view grad_gλ_all[2:end])'
    end

    neg_gλ[1] = -(dot(λ_all, fx) + mapreduce((ai, bi, Δxi) -> ai * Δxi^2 + bi * Δxi, +, a, b, Δx; init=zero(T)))

    return nothing
end
