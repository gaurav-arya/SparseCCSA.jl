"""
This structure contains information about the current primal
iterate, which is sufficient to specify the dual problem.
"""
struct Iterate{T,L}
    x::Vector{T} # (n x 1) x 1 iterate xᵏ
    fx::Vector{T} # (m+1) x 1 values of objective and constraints
    ∇fx::L # (m+1) x n Jacobian linear operator at x
    ρ::Vector{T} # (m+1) x 1 penality weight
    σ::Vector{T} # n x 1 axes lengths of trust region
    lb::Vector{T} # n x 1 lower bounds on solution
    ub::Vector{T} # n x 1 upper bounds on solution
    Δx::Vector{T} # n x 1 xᵏ⁺¹ - xᵏ
    Δx_last::Vector{T} # n x 1 xᵏ - xᵏ⁻¹
    gx::Vector{T} # (m+1) x 1 values of approximate objective and constraints
end

"""
Mutable buffers used by the dual optimization algorithm.
"""
struct DualBuffers{T}
    a::Vector{T} # n x 1 buffer
    b::Vector{T} # n x 1 buffer
    δ::Vector{T} # n x 1 buffer
end

"""
A callable structure for evaluating the dual function and its gradient.
"""
struct DualEvaluator{T, L}
    iterate::Iterate{T, L}
    buffers::DualBuffers{T}
end

"""
    (evaluator::DualEvaluator{T})(∇gλ, gλ, λ)

Set the dual gradient ∇gλ and dual objective gλ
in-place to their new values at λ.
The internal dual buffer δ is also set so that
xᵏ + δ is the optimal primal.
(Note: negated use of g)
"""
function (evaluator::DualEvaluator{T})(∇gλ, λ) where {T}
    @unpack σ, ρ, x, fx, ∇fx, lb, ub = evaluator.iterate
    @unpack a, b, δ = evaluator.buffers
    λ_all = CatView([one(T)], λ)

    @. a = dot(λ_all, ρ) / (2 * σ ^ 2)
    mul!(b, ∇fx', λ_all)
    @. δ = clamp(-b / (2 * a), -σ, σ)
    @. δ = clamp(δ, lb - x, ub - x)
    gλ[1] = dot(λ_all, fx) + sum(@. a * δ^2 + b * δ)
    mul!(∇gλ, ∇fx, δ, zero(T), -one(T))
    @. ∇gλ -= fx + sum(abs2, δ / σ) / 2 * ρ
    return nothing
end

#=
    xtol_rel::T # relative tolerence
    xtol_abs::T # absolute tolerence
    ftol_rel::T # relative tolerence
    ftol_abs::T # absolute tolerence
    max_iters::Int # max number of iterations
=#

struct CCSAOptimizer{T,F,L,D}
    f_and_∇f::F # f(x) = (m+1, (m+1) x n linear operator)
    iterate::Iterate{T,L}
    dual_optimizer::D
    max_iters::Int
    max_inner_iters::Int
end

struct Solution{T}
    x::Vector{T}
    RET::Symbol
end

"""
Return a CCSAOptimizer that can be step!'d. 
Free to allocate here.
"""
function init(f_and_∇f, lb, ub, n, m; x0::Vector{T}, max_iters, max_inner_iters, max_dual_iters, ∇f_prototype)
    x0 === nothing && (x0 = zeros(n))

    iterate = Iterate(x0, zeros(T, m+1), ∇f_prototype, ones(T, m+1), ones(T, n), 
                      lb, ub, zeros(T, n), zeros(T, n), zeros(T, m+1))
    dual_buffers = DualBuffers(zeros(T, n), zeros(T, n), zeros(T, n))
    dual_evaluator = DualEvaluator(iterate, dual_buffers)
    dual_iterate = Iterate(λ, [zero(T)], zeros(T, m), [one(T)], ones(T, m),
                           zeros(typemin(T), m), zeros(T, m), zeros(T, m), zeros(T, m), [zero(T)])
    dual_optimizer = CCSAOptimizer(dual_evaluator, dual_iterate, nothing, max_dual_iters, 0)

    # Initialize objective and gradient 
    f_and_∇f(fx, jac_prototype, x0)
    # Check feasibility
    any(@view fx[2:end] .> 0) && return Solution(x0, :INFEASIBLE)

    return CCSAOptimizer(f_and_∇f, iterate, dual_optimizer, max_iters, max_inner_iters)
end

"""
    step!(optimizer::CCSAOptimizer)

Perform one CCSA iteration.
"""
function step!(optimizer::CCSAOptimizer)
    @unpack f_and_∇f, iterate, dual_optimizer, max_inner_iters = optimizer
    iterate.Δx_last .= iterate.Δx

    # Solve the dual problem, searching for a conservative solution. 
    for i in 1:max_inner_iters
        # Optimize dual problem
        opt!(dual_optimizer) # TODO: handle base case.
        dual_evaluator = dual_optimizer.f_and_∇f
        dual_iterate = dual_optimizer.iterate

        # Run dual evaluator at dual opt and obtain δ
        dual_evaluator(dual_iterate.∇fx, dual_iterate.x) 
        iterate.Δx = dual_evaluator.buffers.δ

        # Check if conservative
        iterate.gx .= iterate.fx .+ sum(abs2, δ ./ iterate.σ) / 2 .* iterate.ρ
        mul!(iterate.gx, iterate.∇fx, δ, true, true)

        f_and_∇f(iterate.∇fx, iterate.x + δ)
        conservative = Iterators.map(>, gx_new, iterate.fx)
        all(conservative) && break
        iterate.ρ[.!conservative] *= 2 # increase ρ until achieving conservative approximation
        dual_iterate.ρ .= one(T) # reinitialize penality weights
        dual_iterate.σ .= one(T) # reinitialize radii of trust region
        dual_iterate.x .= zero(T) # reinitialize starting point of Lagrange multipliers
    end
    
    # Update σ based on monotonicity of changes
    map!((σ, Δx, Δx_last) -> sign(Δx) == sign(Δx_last) ? 2σ : σ/2, iterate.σ, iterate.σ, iterate.Δx, iterate.Δx_last)
    # Halve ρ (be less conservative)
    iterate.ρ /= 2
    iterate.x .+= iterate.Δx
    iterate.Δx_last .= iterate.Δx
    #=
        if norm(opt.Δx, Inf) < opt.xtol_abs
            opt.RET = :XTOL_ABS
            return
        end
        if norm(opt.Δx, Inf) / norm(opt.x, Inf)  < opt.xtol_rel
            opt.RET = :XTOL_REL
            return
        end
        if norm(Δf, Inf) < opt.ftol_abs
            opt.RET = :FTOL_REL
            return
        end
        if norm(Δf, Inf) / norm(f, Inf) < opt.ftol_rel
            opt.RET = :FTOL_REL
            return
        end
    =#
end

function solve!(opt::CCSAOptimizer)
    for i in 1:opt.max_iters
        step!(opt)
    end
    return Solution(opt.iterate.x, :MAX_ITERS)
end



# optimize problem with no constraint
function optimize_simple(opt::CCSAState{T}) where {T}
    monotonic = BitVector(undef, opt.n)
    while opt.iters < opt.max_iters
        opt.fx, opt.∇fx = opt.f_and_∇f(opt.x)
        opt.a .= opt.ρ[1] / 2 ./ (opt.σ) .^ 2
        opt.b .= opt.∇fx[:]
        for i in 1:10
            @. opt.Δx = clamp(-opt.b / (2 * opt.a), -opt.σ, opt.σ)
            @. opt.Δx = clamp(opt.Δx, opt.lb - opt.x, opt.ub - opt.x)
            opt.gλ = opt.fx[1] + sum(@. opt.a * opt.Δx^2 + opt.b * opt.Δx)
            if opt.gλ ≥ opt.f_and_∇f(opt.x + opt.Δx)[1][1] # check conservative
                break
            end
            opt.ρ *= 2
            opt.a *= 2
        end
        opt.ρ /= 2
        monotonic .= signbit.(opt.Δx_last) .== signbit.(opt.Δx) # signbit avoid multiplication
        opt.σ[monotonic] *= 2 # double trust region if xⱼ moves monotomically
        opt.σ[.!monotonic] /= 2 # shrink trust region if xⱼ oscillates
        opt.x .+= opt.Δx
        opt.Δx_last .= opt.Δx
        if norm(opt.Δx, Inf) < opt.xtol_rel
            return
        end
    end
end

