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
function (evaluator::DualEvaluator{T})(gλ, ∇gλ, λ) where {T}
    @unpack σ, ρ, x, fx, ∇fx, lb, ub = evaluator.iterate
    @unpack a, b, δ = evaluator.buffers
    λ_all = CatView([one(T)], λ)
    ∇gλ_all = CatView([one(T)], λ)

    # @info "In evaluator" size(a) size(b) size(∇fx) size(λ_all) size(∇gλ) size(δ) size(fx) size(σ)
    a .= 1 / dot(λ_all, ρ) .* (2 .* σ .^ 2)
    mul!(b, ∇fx', λ_all)
    @. δ = clamp(-b / (2 * a), -σ, σ)
    @. δ = clamp(δ, lb - x, ub - x)
    gλ[1] = dot(λ_all, fx) + sum(@. a * δ^2 + b * δ)
    # Below we populate ∇gλ_all, i.e. the gradient WRT λ_0, ..., λ_m,
    # although we ultimately don't care abuot the first entry.
    mul!(∇gλ_all, ∇fx, δ, zero(T), -one(T))
    ∇gλ_all .-= fx + sum(abs2, δ ./ σ) ./ 2 .* ρ
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
function init(f_and_∇f, lb, ub, n, m; x0::Vector{T}, max_iters, max_inner_iters, max_dual_iters, max_dual_inner_iters, ∇fx_prototype) where {T}
    x0 = (x0 === nothing) ? zeros(n) : copy(x0)
    ∇fx = copy(∇fx_prototype)

    # Setup primal iterate, with n variables and m constraints
    iterate = Iterate(x0, zeros(T, m+1), ∇fx, ones(T, m+1), ones(T, n), 
                      lb, ub, zeros(T, n), zeros(T, n), zeros(T, m+1))

    # Setup dual iterate, with m variables and 0 constraints
    dual_buffers = DualBuffers(zeros(T, n), zeros(T, n), zeros(T, n))
    dual_evaluator = DualEvaluator(iterate, dual_buffers)
    dual_iterate = Iterate(zeros(T, m), [zero(T)], zeros(T, 1, m), [one(T)], ones(T, m),
                           fill(typemin(T), m), zeros(T, m), zeros(T, m), zeros(T, m), [zero(T)])

    # Setup dual dual iterate, with 0 variables and 0 constraints
    dual_dual_evaluator = DualEvaluator(dual_iterate, DualBuffers(zeros(T, m), zeros(T, m), zeros(T, m)))
    dual_dual_iterate = Iterate(T[], [zero(T)], zeros(T, 1, 1), [one(T)], T[], T[], T[], T[], T[], [zero(T)])

    # Setup optimizers
    dual_dual_optimizer = CCSAOptimizer(dual_dual_evaluator, dual_dual_iterate, nothing, 5, 0)
    dual_optimizer = CCSAOptimizer(dual_evaluator, dual_iterate, dual_dual_optimizer, max_dual_iters, max_dual_inner_iters)
    optimizer = CCSAOptimizer(f_and_∇f, iterate, dual_optimizer, max_iters, max_inner_iters)

    # Initialize objective and gradient (TODO: should this move into step! ?)
    f_and_∇f(iterate.fx, iterate.∇fx, iterate.x)

    return optimizer 
end

"""
    step!(optimizer::CCSAOptimizer)

Perform one CCSA iteration.
"""
function step!(optimizer::CCSAOptimizer{T}) where {T}
    @unpack f_and_∇f, iterate, dual_optimizer, max_inner_iters = optimizer
    iterate.Δx_last .= iterate.Δx

    is_primal = length(iterate.x) == 80 && !(f_and_∇f isa DualEvaluator)
    is_dual = length(iterate.x) == 80 && (f_and_∇f isa DualEvaluator)
    if is_primal
        usol = iterate.x[1:40]
        tsol = iterate.x[41:80]
        @info "in step!" maximum(usol - tsol) feasible=all((@view iterate.fx[2:end]) .<= 0) 
    end
    # Check feasibility
    any((@view iterate.fx[2:end]) .> 0) && return Solution(iterate.x, :INFEASIBLE)

    # Solve the dual problem, searching for a conservative solution. 
    for i in 1:max_inner_iters
        #= 
        Optimize dual problem. If dual_optimizer is nothing,
        this means the problem has dimension 0.
        =#
        (dual_optimizer !== nothing) && solve!(dual_optimizer) 
        dual_evaluator = dual_optimizer.f_and_∇f
        dual_iterate = dual_optimizer.iterate

        # Run dual evaluator at dual opt and obtain δ
        dual_evaluator(dual_iterate.fx, dual_iterate.∇fx, dual_iterate.x) 
        δ = dual_evaluator.buffers.δ
        iterate.Δx .= δ 

        # Check if conservative
        iterate.gx .= iterate.fx .+ sum(abs2, δ ./ iterate.σ) / 2 .* iterate.ρ
        mul!(iterate.gx, iterate.∇fx, δ, true, true)

        f_and_∇f(iterate.fx, iterate.∇fx, iterate.x + δ)
        conservative = Iterators.map(>, iterate.gx, iterate.fx)
        dual_iterate.ρ .= one(T) # reinitialize penality weights
        dual_iterate.σ .= one(T) # reinitialize radii of trust region
        dual_iterate.x .= zero(T) # reinitialize starting point of Lagrange multipliers
        dual_evaluator(dual_iterate.fx, dual_iterate.∇fx, dual_iterate.x)
        all(conservative) && break
        iterate.ρ[.!conservative] *= 2 # increase ρ until achieving conservative approximation
        if i == max_inner_iters
            is_primal && "could not find conservative approx for primal"
            is_dual && "could not find conservative approx for dual"
        end
    end
    
    # Update σ based on monotonicity of changes
    map!((σ, Δx, Δx_last) -> sign(Δx) == sign(Δx_last) ? 2σ : σ/2, iterate.σ, iterate.σ, iterate.Δx, iterate.Δx_last)
    # Halve ρ (be less conservative)
    iterate.ρ ./= 2
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