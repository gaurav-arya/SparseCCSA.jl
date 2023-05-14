#=
    xtol_rel::T # relative tolerence
    xtol_abs::T # absolute tolerence
    ftol_rel::T # relative tolerence
    ftol_abs::T # absolute tolerence
    max_iters::Int # max number of iterations
=#

@kwdef mutable struct CCSAOptimizer{T, F, L, D, H}
    f_and_jac::F # f(x) = (m+1, (m+1) x n linear operator)
    iterate::Iterate{T, L}
    buffers::DualBuffers{T}
    dual_optimizer::D
    max_iters::Int
    max_inner_iters::Int
    iter::Int = 1
    history::H = DataFrame()
end

function reinit!(optimizer::CCSAOptimizer{T}; x0=nothing, lb=nothing, ub=nothing) where {T}
    @unpack iterate, dual_optimizer = optimizer

    # initialize lb and ub
    lb !== nothing && (lb .= typemin(T))
    ub !== nothing && (ub .= typemax(T))
    # reinitialize ρ and σ
    iterate.ρ .= one(T) # reinitialize penality weights
    map!(iterate.σ, iterate.lb, iterate.ub) do lb, ub
        (isinf(lb) || isinf(ub)) ? 1.0 : (ub - lb) / 2.0
    end
    # reinitialize starting point
    if (x0 === nothing)
        iterate.x .= zero(T)
    else
        iterate.x .= x0
    end 
    iterate.x_prev .= iterate.x 
    iterate.x_prevprev .= iterate.x 
    # reinitialize function evaluation and Jacobian
    optimizer.f_and_jac(iterate.fx, iterate.jac_fx, iterate.x)
    # recursively reinitalize dual optimizer
    if dual_optimizer !== nothing
        reinit!(dual_optimizer)
    end
end

function propose_Δx!(Δx, optimizer::CCSAOptimizer{T}; verbosity) where {T}
    if optimizer.dual_optimizer !== nothing
        dual_optimizer = optimizer.dual_optimizer
        reinit!(dual_optimizer)
        dual_sol = solve!(dual_optimizer; verbosity=verbosity-1)

        # We can form the dual evaluator with DualEvaluator(; iterate = optimizer.iterate, buffers=optimizer.buffers),
        # but since we have already formed it for the dual optimizer we just retrieve it here.  
        dual_evaluator = dual_optimizer.f_and_jac
        # Run dual evaluator at dual opt and extract Δx from evaluator's buffer
        # Perhaps this isn't actually necessary? i.e. maybe dual_evaluator.buffers.Δx always (?) already has the right thing
        dual_iterate = dual_optimizer.iterate
        dual_evaluator(dual_iterate.fx, dual_iterate.jac_fx, dual_iterate.x)

        Δx .= dual_evaluator.buffers.Δx

        # return dual soln object (used for logging)
        return dual_sol 
    else
        # the "dual dual" problem has 0 variables and 0 contraints, but running it allows us to retrieve the proposed Δx [length m].
        dual_dual_evaluator = DualEvaluator(; iterate = optimizer.iterate,
                                            buffers = optimizer.buffers)
        dual_dual_evaluator(MArray(SA[zero(T)]), SVector{0, T}(), SVector{0, T}())

        Δx .= dual_dual_evaluator.buffers.Δx

        return nothing
    end
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
function init(f_and_jac, lb, ub, n, m; x0::Vector{T}, max_iters, max_inner_iters,
              max_dual_iters, max_dual_inner_iters, jac_prototype) where {T}
    # x0 = (x0 === nothing) ? zeros(n) : copy(x0)

    # Allocate primal iterate, with n variables and m constraints
    iterate = allocate_iterate(; n, m, T, jac_prototype)
    buffers = allocate_buffers(; n, m, T)

    # Setup dual iterate, with m variables and 0 constraints
    dual_evaluator = DualEvaluator(; iterate, buffers)
    dual_iterate = allocate_iterate_for_dual(; m, T)
    dual_buffers = allocate_buffers_for_dual(; m, T)

    # Setup optimizers
    dual_optimizer = CCSAOptimizer(; f_and_jac = dual_evaluator, iterate = dual_iterate,
                                   buffers = dual_buffers,
                                   dual_optimizer = nothing,
                                   max_iters = max_dual_iters,
                                   max_inner_iters = max_dual_inner_iters)
    optimizer = CCSAOptimizer(; f_and_jac, iterate, buffers, dual_optimizer, max_iters,
                              max_inner_iters)

    # Initialize optimizer
    reinit!(optimizer; x0, lb, ub)

    return optimizer
end

"""
    step!(optimizer::AbstractCCSAOptimizer)

Perform one CCSA iteration.

What are the invariants / contracts?
- optimizer.iterate.{fx,jac_fx} come from applying optimizer.f_and_jac at optimizer.iterate.x
- optimizer.dual_optimizer contains a ref to optimizer.iterate, so updating latter implicitly updates the former. 
"""
function step!(optimizer::CCSAOptimizer{T}; verbosity=0) where {T}
    @unpack f_and_jac, iterate, max_inner_iters, dual_optimizer = optimizer

    # Solve the dual problem, searching for a conservative solution. 
    inner_history = verbosity > 0 ? DataFrame() : nothing
    for i in 1:max_inner_iters
        dual_sol = propose_Δx!(iterate.Δx_proposed, optimizer; verbosity)

        # Compute conservative approximation g at proposed point.
        w = sum(abs2, iterate.Δx_proposed ./ iterate.σ) / 2
        iterate.gx_proposed .= iterate.fx .+ iterate.ρ .* w
        mul!(iterate.gx_proposed, iterate.jac_fx, iterate.Δx_proposed, true, true)

        # Compute f at proposed point. 
        iterate.x_proposed .= iterate.x .+ iterate.Δx_proposed
        f_and_jac(iterate.fx_proposed, nothing, iterate.x_proposed)

        # Increase ρ for non-conservative convex approximations.
        conservative = true 
        for i in eachindex(iterate.ρ)
            approx_error = iterate.gx_proposed[i] - iterate.fx_proposed[i]
            conservative &= (approx_error >= 0)
            if approx_error < 0
                iterate.ρ[i] = min(10.0 * iterate.ρ[i], 1.1 * (iterate.ρ[i] - approx_error / w))
            end
        end

        if verbosity > 0
            push!(inner_history, (;dual_iters=dual_sol.iters, dual_obj=-dual_sol.fx[1], 
                                   dual_opt=dual_sol.x[1], 
                                   ρ=copy(iterate.ρ), 
                                   x_proposed=copy(iterate.x_proposed),
                                   Δx_proposed=copy(iterate.Δx_proposed),
                                   conservative=iterate.gx_proposed .> iterate.fx_proposed,
            ))
        end

        # We are guaranteed to have a better optimum once we find a conservative approximation.
        # but even if not conservative, we can check if we have a better optimum by luck, and
        # therefore update our current point a bit more aggressively within the inner iterations,
        # so long as we are still feasible. (Done mostly for consistency with nlopt.) 
        feasible = all(<=(0), iterate.fx_proposed[2:end])
        better_opt = iterate.fx_proposed[1] < iterate.fx[1]
        if feasible && better_opt
            # Update iterate
            iterate.x .= iterate.x_proposed
            f_and_jac(iterate.fx, iterate.jac_fx, iterate.x) # TODO: can avoid this call if we store jac_fx_proposed in prev call
        end

        # Break out if conservative
        if conservative || (i == max_inner_iters) 
            # (!conservative && is_primal) && @info "Could not find conservative approx for $str"
            break
        end
    end

    # Update σ based on monotonicity of changes
    # only do this after the first iteration, similar to nlopt, since this should be a nullop after first update
    if (optimizer.iter > 1)
        for i in eachindex(iterate.σ)
            Δx2 = (iterate.x[i] - iterate.x_prev[i]) * (iterate.x_prev[i] - iterate.x_prevprev[i])
            scaled = (Δx2 < 0 ? 0.7 : (Δx2 > 0 ? 1.2 : 1)) * iterate.σ[i]
            iterate.σ[i] = if isinf(iterate.ub[i]) || isinf(iterate.lb[i])
                scaled
            else
                range = iterate.ub[i] - iterate.lb[i]
                clamp(scaled, 1e-8 * range, 10 * range)
            end
        end
    end

    # Push new x into storage of previous x's
    iterate.x_prevprev .= iterate.x_prev
    iterate.x_prev .= iterate.x

    # Reduce ρ (be less conservative)
    @. iterate.ρ = max(iterate.ρ / 10, 1e-5)

    if verbosity > 0
        push!(optimizer.history, (;ρ=copy(iterate.ρ), σ=copy(iterate.σ), x=copy(iterate.x), fx=copy(iterate.fx), inner_history))
    end

    optimizer.iter += 1 

    #=
        if norm(optimizer.Δx, Inf) < optimizer.xtol_abs
            optimizer.RET = :XTOL_ABS
            return
        end
        if norm(optimizer.Δx, Inf) / norm(optimizer.x, Inf)  < optimizer.xtol_rel
            optimizer.RET = :XTOL_REL
            return
        end
        if norm(Δxf, Inf) < optimizer.ftol_abs
            optimizer.RET = :FTOL_REL
            return
        end
        if norm(Δxf, Inf) / norm(f, Inf) < optimizer.ftol_rel
            optimizer.RET = :FTOL_REL
            return
        end
    =#
end

function solve!(optimizer::CCSAOptimizer; verbosity=0)

    # TODO: catch stopping conditions in opt and exit here
    for i in 1:(optimizer.max_iters)
        step!(optimizer; verbosity)
    end
    return (; x=optimizer.iterate.x, fx=optimizer.iterate.fx, retcode=:MAX_ITERS, iters=optimizer.max_iters)
end
