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

function propose_Δx(optimizer::CCSAOptimizer{T}) where {T}
    if optimizer.dual_optimizer !== nothing
        dual_optimizer = optimizer.dual_optimizer
        solve!(dual_optimizer)
        # Run dual evaluator at dual opt and extract Δx from evaluator's buffer
        dual_evaluator = dual_optimizer.f_and_jac # equivalent to DualEvaluator(; iterate = optimizer.iterate, buffers=optimizer.buffers) 
        dual_iterate = dual_optimizer.iterate 
        dual_evaluator(dual_iterate.fx, dual_iterate.jac, dual_iterate.x)
        return dual_evaluator.buffers.Δx
    else
        # iterate describes a problem with m variables and 0 constraints.
        # buffers are also length m. 
        dual_dual_evaluator = DualEvaluator(; iterate = optimizer.iterate, buffers = optimizer.buffers)
        # problem: we want to fetch an iterate to feed (gλ is length 1, ∇gλ is length 0)
        return dual_dual_evaluator(SA[one(T)], SVector{0,T}(), SVector{0,T}())
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
# TODO: implement init recursively
function init(f_and_jac, lb, ub, n, m; x0::Vector{T}, max_iters, max_inner_iters,
              max_dual_iters, max_dual_inner_iters, jac_prototype) where {T}
    # x0 = (x0 === nothing) ? zeros(n) : copy(x0)

    # Setup primal iterate, with n variables and m constraints
    iterate = init_iterate(; n, m, x0, jac_fx_prototype = copy(jac_prototype), lb, ub)
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

    return optimizer
end

"""
    step!(optimizer::AbstractCCSAOptimizer)

Perform one CCSA iteration.

What are the invariants / contracts?
- optimizer.iterate.{fx,jac} come from applying optimizer.f_and_jac at optimizer.iterate.x
- optimizer.dual_optimizer.f_and_jac contains a ref to optimizer.iterate, so updating latter updates the former. 
"""
function step!(optimizer::CCSAOptimizer{T}) where {T}
    @unpack f_and_jac, iterate, dual_optimizer, max_inner_iters = optimizer
    iterate.Δx_last .= iterate.Δx

    is_primal = !(f_and_jac isa DualEvaluator)
    is_dual = (f_and_jac isa DualEvaluator) && (length(iterate.x) == 2)
    # is_dual = length(iterate.x) == && (f_and_jac isa DualEvaluator)
    if is_primal
        @info "in step! of primal" repr(iterate.x) repr(dual_optimizer.iterate.x) #maximum(usol - tsol) feasible=all((@view iterate.fx[2:end]) .<=
                                   #                                0)
    elseif is_dual
        @info "in step! of dual" repr(iterate.x) repr(dual_optimizer.iterate.x)
    end
    # Check feasibility
    any((@view iterate.fx[2:end]) .> 0) && return Solution(iterate.x, :INFEASIBLE)

    # Solve the dual problem, searching for a conservative solution. 
    for i in 1:max_inner_iters
        # As shown here, the dual_optimizer's evaluator has a reference to our iterate.
        # TODO: aliasing can be a bit trricky, so either document assumption explicitly
        # (that dual_optimizer implicitly depends on CCSAOptimizer's iterate), or change design.
        @assert iterate == dual_optimizer.f_and_jac.iterate

        #= 
        Optimize dual problem. If dual_optimizer is nothing,
        this means the problem has no constraints (the dual problem has 0 constraints). 
        =#
        # Consider the below line in case where we're calling the dual_dual_optimizer (i.e. we're in the dual here). 
        # Why do we do it?

        # Check if conservative
        # TODO: can this be retrieved from dual evaluator?
        # No, because it doesn't do the linear combinations.
        # BUG: need to negate gx since negated by evaluator. No, this is fine...
        iterate.gx .= iterate.fx .+ sum(abs2, Δx ./ iterate.σ) / 2 .* iterate.ρ
        mul!(iterate.gx, iterate.jac, Δx, true, true)
        f_and_jac(iterate.fx2, iterate.jac, iterate.x + Δx)
        conservative = Iterators.map(>=, iterate.gx, iterate.fx2)

        iterate.ρ[.!conservative] *= 2 # increase ρ until achieving conservative approximation

        # Reinitialize dual_iterate (what's changed? iterate's ρ has changed, and thus dual_evaluator.)
        # TODO: should this reiniitalization occur elsewhere? (E.g. as part of solve! ?)
        # Also, should this reiniitalization occur at all?
        dual_iterate.ρ .= one(T) # reinitialize penality weights
        dual_iterate.σ .= one(T) # reinitialize radii of trust region
        dual_iterate.x .= zero(T) # reinitialize starting point of Lagrange multipliers
        dual_evaluator(dual_iterate.fx, dual_iterate.jac, dual_iterate.x)

        if is_primal
            @info "one primal inner iteration:" all(conservative) repr(iterate.gx) repr(iterate.fx) repr(Δx)
            (i == max_inner_iters) && println("could not find conservative approx for primal")
            # if is_dual
            #     @info "could not find conservative approx for dual" norm(Δx) norm(iterate.ρ)
            # end
        elseif is_dual
            @info "one dual inner iteration" all(conservative) repr(iterate.gx) repr(iterate.fx) repr(Δx)
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
    # Initialize objective and gradient (TODO: should this move into step! ?)
    iterate = optimizer.iterate
    optimizer.f_and_jac(iterate.fx, iterate.jac, iterate.x)

    # TODO: catch stopping conditions in opt and exit here
    for i in 1:(optimizer.max_iters)
        step!(optimizer)
    end
    return Solution(optimizer.iterate.x, :MAX_ITERS)
end