#=
    xtol_rel::T # relative tolerence
    xtol_abs::T # absolute tolerence
    ftol_rel::T # relative tolerence
    ftol_abs::T # absolute tolerence
    max_iters::Int # max number of iterations
=#

"""
Interface:

- propose_Δx
"""
@kwdef struct CCSAOptimizer{T, F, L, D}
    f_and_jac::F # f(x) = (m+1, (m+1) x n linear operator)
    iterate::Iterate{T, L}
    buffers::DualBuffers{T}
    dual_optimizer::D
    max_iters::Int
    max_inner_iters::Int
end

function propose_Δx!(Δx, optimizer::CCSAOptimizer{T}) where {T}
    if optimizer.dual_optimizer !== nothing
        dual_optimizer = optimizer.dual_optimizer
        solve!(dual_optimizer)
        # We can form the dual evaluator with DualEvaluator(; iterate = optimizer.iterate, buffers=optimizer.buffers),
        # but since we have already formed it for the dual optimizer we just retrieve it here.  
        dual_evaluator = dual_optimizer.f_and_jac
        # Run dual evaluator at dual opt and extract Δx from evaluator's buffer
        dual_iterate = dual_optimizer.iterate 
        dual_evaluator(dual_iterate.fx, dual_iterate.jac_fx, dual_iterate.x)
        Δx .= dual_evaluator.buffers.Δx 

        # Reinitialize dual_iterate (what's changed? iterate's ρ has changed, and thus dual_evaluator.)
        # TODO: should this reinitalization occur elsewhere? (E.g. as part of solve!, as is already done for x, fx ?)
        # Also, should this reiniitalization occur at all? (could it be useful to "remember" σ/ρ/x used previously?)
        # Yes, I think it should, because ρ and σ seem to become much larger near convergence (while ρ / σ^2 remains same).
        dual_iterate.ρ .= one(T) # reinitialize penality weights
        dual_iterate.σ .= one(T) # reinitialize radii of trust region
        dual_iterate.x .= zero(T) # reinitialize starting point of Lagrange multipliers
        dual_evaluator(dual_iterate.fx, dual_iterate.jac_fx, dual_iterate.x)

        return nothing
    else
        # the "dual dual" problem has 0 variables and 0 contraints, but running it allows us to retrieve the proposed Δx [length m].
        dual_dual_evaluator = DualEvaluator(; iterate = optimizer.iterate, buffers = optimizer.buffers)
        dual_dual_evaluator(MArray(SA[zero(T)]), SVector{0,T}(), SVector{0,T}())
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
    dual_iterate = init_iterate_for_dual(; m, T)
    dual_evaluator = DualEvaluator(; iterate, buffers)
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
function step!(optimizer::CCSAOptimizer{T}) where {T}
    @unpack f_and_jac, iterate, max_inner_iters = optimizer

    is_primal = optimizer.dual_optimizer !== nothing

    # Check feasibility
    # any((@view iterate.fx[2:end]) .> 0) && return Solution(iterate.x, :INFEASIBLE)

    # Solve the dual problem, searching for a conservative solution. 
    for i in 1:max_inner_iters
        #= 
        Optimize dual problem. If dual_optimizer is nothing,
        this means the problem has no constraints (the dual problem has 0 constraints). 
        =#
        propose_Δx!(iterate.Δx_proposed, optimizer)

        # Compute conservative approximation g at proposed point.
        iterate.gx .= iterate.fx .+ sum(abs2, iterate.Δx_proposed ./ iterate.σ) / 2 .* iterate.ρ
        mul!(iterate.gx, iterate.jac_fx, iterate.Δx_proposed, true, true)
        # Compute f at proposed point. TODO: don't need jac here, only f
        iterate.x_proposed .= iterate.x + iterate.Δx_proposed 
        f_and_jac(iterate.fx_proposed, nothing, iterate.x_proposed)
        # Check if conservative
        conservative = Iterators.map(>=, iterate.gx, iterate.fx_proposed)

        if is_primal
            @info "Completed 1 primal inner iteration" repr(iterate.Δx_proposed) repr(iterate.fx_proposed) repr(iterate.gx) repr(iterate.fx) repr(iterate.ρ) repr(collect(conservative)) repr(optimizer.dual_optimizer.iterate.x)
        else
            # @info "completed 1 dual inner iteration" repr(iterate.Δx_proposed) repr(iterate.x) repr(iterate.fx_proposed) repr(iterate.gx) repr(iterate.ρ)
        end

        iterate.ρ[.!conservative] *= 2 # increase ρ for non-conservative convex approximations.

        if all(conservative) || (i == max_inner_iters)
            !all(conservative) && @info "Could not find conservative approx"
            break
        end

    end

    # Update iterate
    iterate.Δx_last .= iterate.Δx
    iterate.Δx .= iterate.Δx_proposed 
    iterate.x .+= iterate.Δx
    f_and_jac(iterate.fx, iterate.jac_fx, iterate.x)
    # Update σ based on monotonicity of changes
    map!((σ, Δx, Δx_last) -> sign(Δx) == sign(Δx_last) ? 2σ : σ / 2, iterate.σ, iterate.σ, iterate.Δx, iterate.Δx_last)
    # Halve ρ (be less conservative)
    iterate.ρ ./= 2

    if is_primal
        @info "Completed 1 primal outer iteration" repr(iterate.x) repr(iterate.ρ) repr(iterate.σ) repr(iterate.fx) repr(iterate.jac_fx) repr(iterate.Δx_last) repr(iterate.Δx)
    else
        # @info "Completed 1 dual outer iteration" repr(iterate.x) repr(iterate.ρ) repr(iterate.σ) repr(iterate.fx) repr(iterate.Δx_last) repr(iterate.Δx)
    end
    #=
        if norm(opt.Δx, Inf) < opt.xtol_abs
            opt.RET = :XTOL_ABS
            return
        end
        if norm(opt.Δx, Inf) / norm(opt.x, Inf)  < opt.xtol_rel
            opt.RET = :XTOL_REL
            return
        end
        if norm(Δxf, Inf) < opt.ftol_abs
            opt.RET = :FTOL_REL
            return
        end
        if norm(Δxf, Inf) / norm(f, Inf) < opt.ftol_rel
            opt.RET = :FTOL_REL
            return
        end
    =#
end

function solve!(optimizer::CCSAOptimizer)

    # TODO: catch stopping conditions in opt and exit here
    for i in 1:(optimizer.max_iters)
        step!(optimizer)
    end
    return Solution(optimizer.iterate.x, :MAX_ITERS)
end