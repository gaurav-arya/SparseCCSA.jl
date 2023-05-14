function reinit!(optimizer::CCSAOptimizer{T}; x0=nothing, lb=nothing, ub=nothing) where {T}
    @unpack cache, dual_optimizer = optimizer

    # Reset optimizer stats 
    reset!(optimizer.stats)
    # Initialize lb and ub
    (lb !== nothing) && (cache.lb .= lb) 
    (ub !== nothing) && (cache.ub .= ub) 
    # Reinitialize ρ and σ
    cache.ρ .= one(T)
    map!(cache.σ, cache.lb, cache.ub) do lb, ub
        (isinf(lb) || isinf(ub)) ? 1.0 : (ub - lb) / 2.0
    end
    # Reinitialize starting point (default: keep what we are already at)
    (x0 !== nothing) && (cache.x .= x0) 
    cache.x_prev .= cache.x 
    cache.x_prevprev .= cache.x 
    # Reinitialize function evaluation and Jacobian
    optimizer.f_and_jac(cache.fx, cache.jac_fx, cache.x)
    # Recursively reinitalize dual optimizer
    if dual_optimizer !== nothing
        reinit!(dual_optimizer; x0=zero(T), lb=zero(T), ub=typemax(T))
    end
end

function propose_Δx!(Δx, optimizer::CCSAOptimizer{T}; verbosity) where {T}
    if optimizer.dual_optimizer !== nothing
        dual_optimizer = optimizer.dual_optimizer
        reinit!(dual_optimizer)
        dual_sol = solve!(dual_optimizer; verbosity=verbosity-1)

        # We can form the dual evaluator with DualEvaluator(; cache = optimizer.cache),
        # but since we have already formed it for the dual optimizer we just retrieve it here.  
        dual_evaluator = dual_optimizer.f_and_jac
        # Run dual evaluator at dual opt and extract Δx from evaluator's buffer
        # Perhaps this isn't actually necessary? i.e. maybe dual_evaluator.cache.Δx always (?) already has the right thing
        dual_cache = dual_optimizer.cache
        dual_evaluator(dual_cache.fx, dual_cache.jac_fx, dual_cache.x)

        Δx .= optimizer.cache.Δx

        # Return dual solution object (used for logging)
        return dual_sol 
    else
        # the "dual dual" problem has 0 variables and 0 contraints, but running it allows us to retrieve the proposed Δx [length m].
        dual_dual_evaluator = DualEvaluator(; cache = optimizer.cache)
        dual_dual_evaluator(MArray(SA[zero(T)]), SVector{0, T}(), SVector{0, T}())

        Δx .= optimizer.cache.Δx

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
function init(f_and_jac, n, m, T, jac_prototype; lb=nothing, ub=nothing, x0=nothing, 
              max_iters=nothing, max_inner_iters=nothing, ftol_abs=nothing, ftol_rel=nothing,
              max_dual_iters=nothing, max_dual_inner_iters=nothing, dual_ftol_abs=nothing,
              dual_ftol_rel=nothing)

    # Allocate primal cache, with n variables and m constraints
    cache = allocate_cache(; n, m, T, jac_prototype)

    # Allocate dual cache, with m variables and 0 constraints
    dual_cache = allocate_cache_for_dual(; m, T)

    # Setup optimizers
    dual_optimizer = CCSAOptimizer(; f_and_jac = DualEvaluator(; cache), cache = dual_cache,
                                   dual_optimizer = nothing,
                                   settings = CCSASettings(; max_iters = max_dual_iters, 
                                                           max_inner_iters = max_dual_inner_iters,
                                                           ftol_abs = dual_ftol_abs, ftol_rel = dual_ftol_rel))
    optimizer = CCSAOptimizer(; f_and_jac, cache, dual_optimizer,
                                settings = CCSASettings(; max_iters, max_inner_iters, ftol_abs, ftol_rel))

    # Initialize optimizer
    x0 = (x0 === nothing) ? zeros(T, n) : copy(x0)
    lb = (lb === nothing) ? fill(typemin(T), n) : lb 
    ub = (ub === nothing) ? fill(typemax(T), n) : ub 
    reinit!(optimizer; x0, lb, ub)

    return optimizer
end

"""
    step!(optimizer::AbstractCCSAOptimizer)

Perform one CCSA iteration.

What are the invariants / contracts?
- optimizer.cache.{fx,jac_fx} come from applying optimizer.f_and_jac at optimizer.cache.x
- optimizer.dual_optimizer contains a ref to optimizer.cache, so updating latter implicitly updates the former. 
"""
function step!(optimizer::CCSAOptimizer{T}; verbosity=0) where {T}
    @unpack f_and_jac, cache, dual_optimizer, stats, settings = optimizer

    retcode = get_retcode(optimizer)
    (retcode != :CONTINUE) && return retcode

    if optimizer.stats.outer_iters_done > 0
        # Push new x into storage of previous x's
        cache.x_prevprev .= cache.x_prev
        cache.x_prev .= cache.x
        # Push new fx[1] into storage of previous fx[1]
        cache.fx_prev .= cache.fx
    end

    # Solve the dual problem, searching for a conservative solution. 
    inner_history = verbosity > 1 ? DataFrame() : nothing
    while true 
        dual_sol = propose_Δx!(cache.Δx_proposed, optimizer; verbosity)

        # Compute conservative approximation g at proposed point.
        w = sum(abs2, cache.Δx_proposed ./ cache.σ) / 2
        cache.gx_proposed .= cache.fx .+ cache.ρ .* w
        mul!(cache.gx_proposed, cache.jac_fx, cache.Δx_proposed, true, true)

        # Compute f at proposed point. 
        cache.x_proposed .= cache.x .+ cache.Δx_proposed
        f_and_jac(cache.fx_proposed, cache.jac_fx_proposed, cache.x_proposed)

        # Increase ρ for non-conservative convex approximations.
        conservative = true 
        for i in eachindex(cache.ρ)
            approx_error = cache.gx_proposed[i] - cache.fx_proposed[i]
            conservative &= (approx_error >= 0)
            if approx_error < 0
                cache.ρ[i] = min(10.0 * cache.ρ[i], 1.1 * (cache.ρ[i] - approx_error / w))
            end
        end

        if verbosity > 1
            push!(inner_history, (;dual_iters=dual_sol.stats.outer_iters_done, dual_obj=-dual_sol.fx[1], 
                                   dual_opt=dual_sol.x[1], 
                                   ρ=copy(cache.ρ), 
                                   x_proposed=copy(cache.x_proposed),
                                   Δx_proposed=copy(cache.Δx_proposed),
                                   conservative=cache.gx_proposed .> cache.fx_proposed,
            ))
        end

        # We are guaranteed to have a better optimum once we find a conservative approximation.
        # but even if not conservative, we can check if we have a better optimum by luck, and
        # therefore update our current point a bit more aggressively within the inner iterations,
        # so long as we are still feasible. (Done mostly for consistency with nlopt.) 
        feasible = all(<=(0), cache.fx_proposed[2:end])
        better_opt = cache.fx_proposed[1] < cache.fx[1]
        inner_done = conservative || (stats.inner_iters_cur_done == settings.max_inner_iters) 
        if feasible && (better_opt || inner_done)
            # Update cache
            cache.x .= cache.x_proposed
            cache.fx .= cache.fx_proposed
            cache.jac_fx .= cache.jac_fx_proposed
        end

        stats.inner_iters_cur_done += 1
        stats.inner_iters_done += 1

        # Break out if conservative
        inner_done && break
    end

    optimizer.stats.outer_iters_done += 1

    # Update σ based on monotonicity of changes
    # only do this after the first iteration, similar to nlopt, since this should be a nullop after first update
    if optimizer.stats.outer_iters_done > 1
        for i in eachindex(cache.σ)
            Δx2 = (cache.x[i] - cache.x_prev[i]) * (cache.x_prev[i] - cache.x_prevprev[i])
            scaled = (Δx2 < 0 ? 0.7 : (Δx2 > 0 ? 1.2 : 1)) * cache.σ[i]
            cache.σ[i] = if isinf(cache.ub[i]) || isinf(cache.lb[i])
                scaled
            else
                range = cache.ub[i] - cache.lb[i]
                clamp(scaled, 1e-8 * range, 10 * range)
            end
        end
    end

    # Reduce ρ (be less conservative)
    @. cache.ρ = max(cache.ρ / 10, 1e-5)

    if verbosity > 0
        push!(stats.history, (;ρ=copy(cache.ρ), σ=copy(cache.σ), x=copy(cache.x), fx=copy(cache.fx), inner_history))
    end

    return retcode
end

function get_retcode(optimizer::CCSAOptimizer)
    @unpack cache, stats, settings = optimizer

    if (settings.max_iters !== nothing) && (stats.outer_iters_done >= settings.max_iters)
        return :MAX_ITERS
    end

    if (stats.outer_iters_done > 1)
        # Objective tolerance 
        Δfx = abs(cache.fx[1] - cache.fx_prev[1])  
        if (settings.ftol_abs !== nothing) && (Δfx < settings.ftol_abs)
            return :FTOL_ABS
        elseif (settings.ftol_rel !== nothing) && (abs(Δfx / cache.fx[1]) < settings.ftol_rel)
            return :FTOL_REL
        end
    end

    return :CONTINUE
end

function solve!(optimizer::CCSAOptimizer; verbosity=0)
    retcode = :CONTINUE
    while retcode == :CONTINUE
        retcode = step!(optimizer; verbosity)
    end
    return (; x=optimizer.cache.x, fx=optimizer.cache.fx, retcode, stats=optimizer.stats)
end
