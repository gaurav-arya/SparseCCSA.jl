#=
    xtol_rel::T # relative tolerence
    xtol_abs::T # absolute tolerence
    ftol_rel::T # relative tolerence
    ftol_abs::T # absolute tolerence
    max_iters::Int # max number of iterations
=#

@kwdef struct CCSAOptimizer{T, F, L, D}
    f_and_jac::F # f(x) = (m+1, (m+1) x n linear operator)
    iterate::Iterate{T, L}
    buffers::DualBuffers{T}
    dual_optimizer::D
    max_iters::Int
    max_inner_iters::Int
end

function reinit!(optimizer::CCSAOptimizer{T}) where {T}
    iterate = optimizer.iterate

    iterate.ρ .= one(T) # reinitialize penality weights
    iterate.σ .= one(T) # reinitialize radii of trust region
    iterate.x .= zero(T) # reinitialize starting point of Lagrange multipliers
    iterate.Δx .= zero(T)
    iterate.Δx_last .= zero(T)
    optimizer.f_and_jac(iterate.fx, iterate.jac_fx, iterate.x)
end

function propose_Δx!(Δx, optimizer::CCSAOptimizer{T}; verbose=false) where {T}
    if optimizer.dual_optimizer !== nothing
        dual_optimizer = optimizer.dual_optimizer
        reinit!(dual_optimizer)
        sol = solve!(dual_optimizer)
        if (verbose)
		    @printf "CCSA dual converged in %d iters to g=%g:\n" sol.iters sol.fx[1]
		    # for i in 1:m 
            #     @printf "    CCSA y[%u]=%g, gc[%u]=%g\n" i y[i] i dd.gcval[i]
            # end
        end
        # We can form the dual evaluator with DualEvaluator(; iterate = optimizer.iterate, buffers=optimizer.buffers),
        # but since we have already formed it for the dual optimizer we just retrieve it here.  
        dual_evaluator = dual_optimizer.f_and_jac
        # Run dual evaluator at dual opt and extract Δx from evaluator's buffer
        dual_iterate = dual_optimizer.iterate
        dual_evaluator(dual_iterate.fx, dual_iterate.jac_fx, dual_iterate.x)
        # @show dual_iterate.x sol.x

        Δx .= dual_evaluator.buffers.Δx

        return nothing
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

    # Setup primal iterate, with n variables and m constraints
    iterate = init_iterate(; n, m, x0, jac_prototype, lb, ub)
    buffers = init_buffers(; n, m, T)

    # Setup dual iterate, with m variables and 0 constraints
    dual_evaluator = DualEvaluator(; iterate, buffers)
    dual_iterate = init_iterate_for_dual(; m, T)
    dual_buffers = init_buffers_for_dual(; m, T)

    # Setup optimizers
    dual_optimizer = CCSAOptimizer(; f_and_jac = dual_evaluator, iterate = dual_iterate,
                                   buffers = dual_buffers,
                                   dual_optimizer = nothing,
                                   max_iters = max_dual_iters,
                                   max_inner_iters = max_dual_inner_iters)
    optimizer = CCSAOptimizer(; f_and_jac, iterate, buffers, dual_optimizer, max_iters,
                              max_inner_iters)

    # Initialize objective and gradient
    # TODO: could be acceptable to move this to beginning of step!
    f_and_jac(iterate.fx, iterate.jac_fx, iterate.x)
    # @show dual_iterate.fx dual_iterate.jac_fx dual_iterate.x
    dual_evaluator(dual_iterate.fx, dual_iterate.jac_fx, dual_iterate.x)

    return optimizer
end

"""
    step!(optimizer::AbstractCCSAOptimizer)

Perform one CCSA iteration.

What are the invariants / contracts?
- optimizer.iterate.{fx,jac_fx} come from applying optimizer.f_and_jac at optimizer.iterate.x
- optimizer.dual_optimizer contains a ref to optimizer.iterate, so updating latter implicitly updates the former. 
"""
function step!(optimizer::CCSAOptimizer{T}; verbose=false) where {T}
    @unpack f_and_jac, iterate, max_inner_iters = optimizer

    is_primal = optimizer.dual_optimizer !== nothing
    str = is_primal ? "primal" : "dual"
    # is_primal &&
    #     @info "Starting $str outer iteration" repr(iterate.x) repr(iterate.ρ) repr(iterate.σ) repr(iterate.jac_fx)

    # Check feasibility
    # any((@view iterate.fx[2:end]) .> 0) && return Solution(iterate.x, :INFEASIBLE)

    # Solve the dual problem, searching for a conservative solution. 
    for i in 1:max_inner_iters
        propose_Δx!(iterate.Δx_proposed, optimizer; verbose)

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
            conservative &= (approx_error > 0)
            if approx_error < 0
                iterate.ρ[i] = min(10.0 * iterate.ρ[i], 1.1 * (iterate.ρ[i] - approx_error / w))
            end
        end

        if verbose
		    @printf "CCSA inner iteration\n"
            for i in 1:length(iterate.ρ)
                @printf "                CCSA rho[%u] -> %g\n" i iterate.ρ[i];
                @printf "                CCSA conservative[%u] -> %g\n" i iterate.gx_proposed[i] > iterate.fx_proposed[i];
            end
        end

        # Break out if conservative
        if conservative || (i == max_inner_iters) 
            # (!conservative && is_primal) && @info "Could not find conservative approx for $str"
            break
        end
    end

    # Update iterate
    iterate.Δx_last .= iterate.Δx
    iterate.Δx .= iterate.Δx_proposed
    iterate.x .= iterate.x_proposed
    f_and_jac(iterate.fx, iterate.jac_fx, iterate.x)

    # Update σ based on monotonicity of changes
    for i in eachindex(iterate.σ)
        scaled = (sign(iterate.Δx[i]) == sign(iterate.Δx_last[i]) ? 1.2 : 0.7) * iterate.σ[i]
        iterate.σ[i] = if isinf(iterate.ub[i]) || isinf(iterate.lb[i])
            scaled
        else
            range = iterate.ub[i] - iterate.lb[i]
            clamp(scaled, 1e-8 * range, 10 * range)
        end
    end

    # Reduce ρ (be less conservative)
    @. iterate.ρ = max(iterate.ρ / 10, 1e-5)

    if verbose
        @printf "CCSA outer iteration: rho -> %g\n" iterate.ρ[1];
        for i in 2:length(iterate.ρ)
            @printf "                CCSA rhoc[%u] -> %g\n" i iterate.ρ[i];
        end
    end

    # is_primal &&
    #     @info "Completed 1 $str outer iteration" repr(iterate.x) repr(iterate.ρ) repr(iterate.σ) repr(iterate.fx) repr(iterate.jac_fx) repr(iterate.Δx_last) repr(iterate.Δx)
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

function solve!(optimizer::CCSAOptimizer; verbose=false)

    # TODO: catch stopping conditions in opt and exit here
    for i in 1:(optimizer.max_iters)
        step!(optimizer; verbose)
    end
    return (; x=optimizer.iterate.x, fx=optimizer.iterate.fx, retcode=:MAX_ITERS, iters=optimizer.max_iters)
end
