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
    Δx::Vector{T} # n x 1 xᵏ⁺¹ - xᵏ
    Δx_last::Vector{T} # n x 1 xᵏ - xᵏ⁻¹
    gx::Vector{T} # (m+1) x 1 values of approximate objective and constraints
end

function init_iterate(; n, m, x0::Vector{T}, ∇fx_prototype, lb, ub) where {T}
    return Iterate(; x = x0, fx = zeros(T, m + 1), ∇fx = ∇fx_prototype, ρ = ones(T, m + 1),
                   σ = ones(T, n),
                   lb, ub, Δx = zeros(T, n), Δx_last = zeros(T, n), gx = zeros(T, m + 1))
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

    is_dual = (length(λ) > 0)
    if is_dual
        @info "In dual evaluator" repr(λ) repr(x) repr(fx)
    end

    # @info "In evaluator" size(a) size(b) size(∇fx) size(λ_all) size(∇gλ) size(δ) size(fx) size(σ)
    a .= dot(λ_all, ρ) ./ (2 .* σ .^ 2)
    mul!(b, ∇fx', λ_all)
    @. δ = clamp(-b / (2 * a), -σ, σ)
    @. δ = clamp(δ, lb - x, ub - x)
    @show a b δ
    # @info "in evaluator" norm(ρ) norm(δ)
    gλ[1] = dot(λ_all, fx) + sum(@. a * δ^2 + b * δ)
    # Below we populate ∇gλ_all, i.e. the gradient WRT λ_0, ..., λ_m,
    # although we ultimately don't care abuot the first entry.
    mul!(∇gλ_all, ∇fx, δ)
    ∇gλ_all .+= fx + sum(abs2, δ ./ σ) ./ 2 .* ρ

    # Negate to turn into minimization problem
    # gλ .*= -1
    # ∇gλ .*= -1

    # if is_dual
    #     @info "End of dual evaluator" repr(λ) repr(x) repr(fx) repr(δ) repr(gλ) repr(∇gλ_all)
    # end
    return nothing
end

#=
    xtol_rel::T # relative tolerence
    xtol_abs::T # absolute tolerence
    ftol_rel::T # relative tolerence
    ftol_abs::T # absolute tolerence
    max_iters::Int # max number of iterations
=#

@kwdef struct CCSAOptimizer{T, F, L, D}
    f_and_∇f::F # f(x) = (m+1, (m+1) x n linear operator)
    iterate::Iterate{T, L}
    dual_optimizer::D
    max_iters::Int
    max_inner_iters::Int
end

@kwdef struct Solution{T}
    x::Vector{T}
    RET::Symbol
end

"""
Return a CCSAOptimizer that can be step!'d. 
Free to allocate here.
"""
# TODO: defaults for kwargs below
# TODO: implement init recursively
function init(f_and_∇f, lb, ub, n, m; x0::Vector{T}, max_iters, max_inner_iters,
              max_dual_iters, max_dual_inner_iters, ∇fx_prototype) where {T}
    # x0 = (x0 === nothing) ? zeros(n) : copy(x0)

    # Setup primal iterate, with n variables and m constraints
    iterate = init_iterate(; n, m, x0, ∇fx_prototype = copy(∇fx_prototype), lb, ub)

    # Setup dual iterate, with m variables and 0 constraints
    dual_evaluator = DualEvaluator(; iterate, buffers = init_buffers(; T, n))
    dual_iterate = init_iterate_for_dual(; m, T)

    # Setup dual dual iterate, with 0 variables and 0 constraints
    dual_dual_evaluator = DualEvaluator(; iterate = dual_iterate,
                                        buffers = init_buffers(; T, n=m))
    dual_dual_iterate = init_iterate_for_dual(; m = 0, T)

    # Setup optimizers
    dual_dual_optimizer = CCSAOptimizer(; f_and_∇f = dual_dual_evaluator,
                                        iterate = dual_dual_iterate,
                                        dual_optimizer = nothing, max_iters = 5,
                                        max_inner_iters = 0)
    dual_optimizer = CCSAOptimizer(; f_and_∇f = dual_evaluator, iterate = dual_iterate,
                                   dual_optimizer = dual_dual_optimizer,
                                   max_iters = max_dual_iters,
                                   max_inner_iters = max_dual_inner_iters)
    optimizer = CCSAOptimizer(; f_and_∇f, iterate, dual_optimizer, max_iters,
                              max_inner_iters)

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

    is_primal = !(f_and_∇f isa DualEvaluator)
    is_dual = (f_and_∇f isa DualEvaluator) && (length(iterate.x) == 2)
    # is_dual = length(iterate.x) == && (f_and_∇f isa DualEvaluator)
    if is_primal
        @info "in step! of primal" iterate.x #maximum(usol - tsol) feasible=all((@view iterate.fx[2:end]) .<=
                                   #                                0)
    elseif is_dual
        @info "in step! of dual" iterate.x
    end
    # Check feasibility
    any((@view iterate.fx[2:end]) .> 0) && return Solution(iterate.x, :INFEASIBLE)

    # Solve the dual problem, searching for a conservative solution. 
    for i in 1:max_inner_iters
        if is_primal
            # dual_evaluator = dual_optimizer.f_and_∇f
            # dual_iterate = dual_optimizer.iterate
            # prev_∇fx = dual_iterate.∇fx
            # dual_evaluator(dual_iterate.fx, dual_iterate.∇fx, dual_iterate.x)
            # δ = dual_evaluator.buffers.δ
            # @info "testing gradient of dual" repr(dual_iterate.fx) repr(dual_iterate.∇fx) repr(prev_∇fx) repr(dual_iterate.x) repr(δ)
            # error("end")
        end

        #= 
        Optimize dual problem. If dual_optimizer is nothing,
        this means the problem has dimension 0.
        =#
        (dual_optimizer !== nothing) && solve!(dual_optimizer)
        dual_evaluator = dual_optimizer.f_and_∇f
        dual_iterate = dual_optimizer.iterate

        # Run dual evaluator at dual opt and obtain δ
        dual_evaluator(dual_iterate.fx, dual_iterate.∇fx, dual_iterate.x)
        δ = dual_evaluator.buffers.δ # problem: this is 0
        iterate.Δx .= δ

        # Check if conservative
        iterate.gx .= iterate.fx .+ sum(abs2, δ ./ iterate.σ) / 2 .* iterate.ρ
        mul!(iterate.gx, iterate.∇fx, δ, true, true)

        f_and_∇f(iterate.fx, iterate.∇fx, iterate.x + δ)
        conservative = Iterators.map(>=, iterate.gx, iterate.fx)
        if is_primal
            @info "one primal inner iteration:" all(conservative) repr(iterate.gx) repr(iterate.fx) repr(δ)
        elseif is_dual
            @info "one dual inner iteration" all(conservative) repr(iterate.gx) repr(iterate.fx) repr(δ)
        end
        break

        dual_iterate.ρ .= one(T) # reinitialize penality weights
        dual_iterate.σ .= one(T) # reinitialize radii of trust region
        dual_iterate.x .= zero(T) # reinitialize starting point of Lagrange multipliers
        dual_evaluator(dual_iterate.fx, dual_iterate.∇fx, dual_iterate.x)

        iterate.ρ[.!conservative] *= 2 # increase ρ until achieving conservative approximation
        if i == max_inner_iters
            is_primal && println("could not find conservative approx for primal")
            # if is_dual
            #     @info "could not find conservative approx for dual" norm(δ) norm(iterate.ρ)
            # end
        end
    end

    # Update σ based on monotonicity of changes
    map!((σ, Δx, Δx_last) -> sign(Δx) == sign(Δx_last) ? 2σ : σ / 2, iterate.σ, iterate.σ,
         iterate.Δx, iterate.Δx_last)
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
    # TODO: catch stopping conditions in opt and exit here
    for i in 1:(opt.max_iters)
        step!(opt)
    end
    return Solution(opt.iterate.x, :MAX_ITERS)
end
