"""
This structure contains information about the current primal
iterate, which is sufficient to specify the dual problem.
"""
@kwdef struct Iterate{T, L}
    x::Vector{T} # (n x 1) x 1 iterate xᵏ
    fx::Vector{T} # (m+1) x 1 values of objective and constraints
    ∇fx::L # (m+1) x n Jacobian linear operator at x
    ρ::Vector{T} # (m+1) x 1 penality weight
    σ::Vector{T} # n x 1 axes lengths of trust region
    lb::Vector{T} # n x 1 lower bounds on solution
    ub::Vector{T} # n x 1 upper bounds on solution
    # Below are buffers used by inner iteration logic. TODO: move into separate struct?
    Δx::Vector{T} # n x 1 xᵏ⁺¹ - xᵏ
    Δx_last::Vector{T} # n x 1 xᵏ - xᵏ⁻¹
    gx::Vector{T} # (m+1) x 1 values of approximate objective and constraints
    fx2::Vector{T} # (m+1) x 1 values of approximate objective and constraints
end

function init_iterate(; n, m, x0::Vector{T}, ∇fx_prototype, lb, ub) where {T}
    return Iterate(; x = x0, fx = zeros(T, m + 1), ∇fx = ∇fx_prototype, ρ = ones(T, m + 1),
                   σ = ones(T, n),
                   lb, ub, Δx = zeros(T, n), Δx_last = zeros(T, n), gx = zeros(T, m + 1), fx2 = zeros(T, m + 1))
end

"""
Instantiates the iterate structure for a dual problem with m constraints.
"""
function init_iterate_for_dual(; m, T)
    return init_iterate(; n = m, m = 0, x0 = zeros(T, m),
                        ∇fx_prototype = zeros(T, 1, m), lb = zeros(m),
                        ub = fill(typemax(T), m))
end

"""
Mutable buffers used by the dual optimization algorithm.
"""
@kwdef struct DualBuffers{T}
    a::Vector{T} # n x 1 buffer
    b::Vector{T} # n x 1 buffer
    δ::Vector{T} # n x 1 buffer
end

function init_buffers(; T, n)
    DualBuffers(zeros(T, n), zeros(T, n), zeros(T, n))
end

"""
A callable structure for evaluating the dual function and its gradient.
"""
@kwdef struct DualEvaluator{T, L}
    iterate::Iterate{T, L}
    buffers::DualBuffers{T}
end

"""
    (evaluator::DualEvaluator{T})(gλ, ∇gλ, λ)

Set the dual gradient ∇gλ and dual objective gλ
in-place to their new values at λ.
The internal dual buffer δ is also set so that
xᵏ + δ is the optimal primal.
Shapes: ∇gλ and λ are m-length, gλ is 1-length.
(Note: negated use of g)
"""
function (evaluator::DualEvaluator{T})(gλ, ∇gλ, λ) where {T}
    @unpack σ, ρ, x, fx, ∇fx, lb, ub = evaluator.iterate
    @unpack a, b, δ = evaluator.buffers
    λ_all = CatView([one(T)], λ)
    ∇gλ_all = CatView([one(T)], ∇gλ)

    a .= dot(λ_all, ρ) ./ (2 .* σ .^ 2)
    mul!(b, ∇fx', λ_all)
    @. δ = clamp(-b / (2 * a), -σ, σ)
    @. δ = clamp(δ, lb - x, ub - x)
    gλ[1] = dot(λ_all, fx) + sum(@. a * δ^2 + b * δ)
    # Below we populate ∇gλ_all, i.e. the gradient WRT λ_0, ..., λ_m,
    # although we ultimately don't care abuot the first entry.
    mul!(∇gλ_all, ∇fx, δ)
    ∇gλ_all .+= fx + sum(abs2, δ ./ σ) ./ 2 .* ρ

    # Negate to turn into minimization problem
    gλ .*= -1
    ∇gλ .*= -1

    return nothing
end