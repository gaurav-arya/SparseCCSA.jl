"""
This structure contains information about the current primal
iterate, which is sufficient to specify the dual problem.
"""
@kwdef struct Iterate{T, L}
    x::Vector{T} # (n x 1) x 1 iterate xᵏ
    fx::Vector{T} # (m+1) x 1 values of objective and constraints
    jac_fx::L # (m+1) x n Jacobian linear operator at x
    ρ::Vector{T} # (m+1) x 1 penality weight
    σ::Vector{T} # n x 1 axes lengths of trust region
    lb::Vector{T} # n x 1 lower bounds on solution
    ub::Vector{T} # n x 1 upper bounds on solution
    # Below are buffers used by inner iteration logic. TODO: move into separate struct, defined in optimize.jl?
    Δx::Vector{T} # n x 1 xᵏ⁺¹ - xᵏ
    Δx_last::Vector{T} # n x 1 xᵏ - xᵏ⁻¹
    gx::Vector{T} # (m+1) x 1 values of approximate objective and constraints
    fx2::Vector{T} # (m+1) x 1 values of approximate objective and constraints
end

function init_iterate(; n, m, x0::Vector{T}, jac_prototype, lb, ub) where {T}
    return Iterate(; x = x0, fx = zeros(T, m + 1), jac_fx = jac_prototype, ρ = ones(T, m + 1),
                   σ = ones(T, n), lb, ub, Δx = zeros(T, n), Δx_last = zeros(T, n), 
                   gx = zeros(T, m + 1), fx2 = zeros(T, m + 1))
end

"""
Instantiates the iterate structure for a dual problem with m constraints.
"""
function init_iterate_for_dual(; m, T)
    return init_iterate(; n = m, m = 0, x0 = zeros(T, m),
                        jac_prototype = zeros(T, 1, m), lb = zeros(m),
                        ub = fill(typemax(T), m))
end

"""
Mutable buffers used by the dual optimization algorithm.
"""
@kwdef struct DualBuffers{T}
    a::Vector{T} # n x 1 buffer
    b::Vector{T} # n x 1 buffer
    Δx::Vector{T} # n x 1 buffer
    λ_all::Vector{T} # (m + 1) x 1 buffer
    grad_gλ_all::Vector{T} # (m + 1) x 1 buffer
end

init_buffers(; n, m, T) = DualBuffers(zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, m + 1), zeros(T, m + 1))
init_buffers_for_dual(; m, T) = init_buffers(; n = m, m = 0, T)

"""
A callable structure for evaluating the dual function and its gradient.
"""
@kwdef struct DualEvaluator{T, L}
    iterate::Iterate{T, L}
    buffers::DualBuffers{T}
end

"""
    (evaluator::DualEvaluator{T})(neg_gλ, neg_grad_gλ, λ)

Set the negated dual gradient neg_grad_gλ [length = m] and negated dual objective neg_gλ [length = 1]
in-place to their new values at λ [length = m].
The evaluator's internal buffer Δx is also set so that xᵏ + Δx is the optimal primal.
"""
function (evaluator::DualEvaluator{T})(neg_gλ, neg_grad_gλ, λ) where {T}
    @unpack σ, ρ, x, fx, jac_fx, lb, ub = evaluator.iterate
    @unpack a, b, Δx, λ_all, grad_gλ_all = evaluator.buffers
    # The dual evaluation turns out to be simpler to express with λ_{1...m} left-augmented by λ_0 = 1.
    # We have special (m+1)-length buffers for this, which is a little wasteful. 
    λ_all[1] = 1 
    λ_all[2:end] .= λ
    grad_gλ_all .= 0

    a .= dot(λ_all, ρ) ./ (2 .* σ .^ 2)
    mul!(b, jac_fx', λ_all)
    @. Δx = clamp(-b / (2 * a), -σ, σ)
    @. Δx = clamp(Δx, lb - x, ub - x)
    neg_gλ[1] = -1 * (dot(λ_all, fx) + sum(@. a * Δx^2 + b * Δx))
    # Below we populate grad_gλ_all, i.e. the gradient WRT λ_0, ..., λ_m,
    # although we ultimately don't care about the first entry.
    mul!(grad_gλ_all, jac_fx, Δx)
    grad_gλ_all .+= fx + sum(abs2, Δx ./ σ) ./ 2 .* ρ

    # Negate to turn into minimization problem
    neg_grad_gλ .= -1 * @view grad_gλ_all[2:end]

    return nothing
end