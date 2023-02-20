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

function propose_Δx!(Δx, optimizer::CCSAOptimizer{T}) where {T}
    if optimizer.dual_optimizer !== nothing
        dual_optimizer = optimizer.dual_optimizer
        reinit!(dual_optimizer)
        sol = solve!(dual_optimizer)
        # We can form the dual evaluator with DualEvaluator(; iterate = optimizer.iterate, buffers=optimizer.buffers),
        # but since we have already formed it for the dual optimizer we just retrieve it here.  
        dual_evaluator = dual_optimizer.f_and_jac
        # Run dual evaluator at dual opt and extract Δx from evaluator's buffer
        dual_iterate = dual_optimizer.iterate 
        dual_evaluator(dual_iterate.fx, dual_iterate.jac_fx, dual_iterate.x)
        @show dual_iterate.x sol.x

        Δx .= dual_evaluator.buffers.Δx 

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
    str = is_primal ? "primal" : "dual" 
    is_primal && @info "Starting $str outer iteration" repr(iterate.x) repr(iterate.ρ) repr(iterate.σ) repr(iterate.jac_fx) 

    # Check feasibility
    # any((@view iterate.fx[2:end]) .> 0) && return Solution(iterate.x, :INFEASIBLE)

    # Solve the dual problem, searching for a conservative solution. 
    for i in 1:max_inner_iters
        propose_Δx!(iterate.Δx_proposed, optimizer)
        # is_primal && (@show solve!(optimizer.dual_optimizer))
        # is_primal && error("e")

        # Compute conservative approximation g at proposed point.
        iterate.gx_proposed .= iterate.fx .+ sum(abs2, iterate.Δx_proposed ./ iterate.σ) / 2 .* iterate.ρ
        mul!(iterate.gx_proposed, iterate.jac_fx, iterate.Δx_proposed, true, true)
        # Compute f at proposed point. 
        iterate.x_proposed .= iterate.x + iterate.Δx_proposed 
        f_and_jac(iterate.fx_proposed, nothing, iterate.x_proposed)
        # Check if conservative
        conservative = Iterators.map(>=, iterate.gx_proposed, iterate.fx_proposed)

        is_primal && @info "Completed 1 $str inner iteration" repr(iterate.Δx_proposed) repr(iterate.x) repr(iterate.x_proposed) repr(iterate.fx_proposed) repr(iterate.gx_proposed) repr(iterate.fx) repr(iterate.ρ) repr(collect(conservative))

        # Increase ρ for non-conservative convex approximations.
        iterate.ρ[.!conservative] .= min(iterate.ρ[.!conservative], )

        if all(conservative) || (i == max_inner_iters)
            !all(conservative) && @info "Could not find conservative approx for $str"
            break
        end

    end

    # Update iterate
    iterate.Δx_last .= iterate.Δx
    iterate.Δx .= iterate.Δx_proposed 
    iterate.x .= iterate.x_proposed
    f_and_jac(iterate.fx, iterate.jac_fx, iterate.x)
    # Update σ based on monotonicity of changes
    map!((σ, Δx, Δx_last) -> sign(Δx) == sign(Δx_last) ? 2σ : σ / 2, iterate.σ, iterate.σ, iterate.Δx, iterate.Δx_last)
    # Halve ρ (be less conservative)
    iterate.ρ ./= 2

    is_primal && @info "Completed 1 $str outer iteration" repr(iterate.x) repr(iterate.ρ) repr(iterate.σ) repr(iterate.fx) repr(iterate.jac_fx) repr(iterate.Δx_last) repr(iterate.Δx)
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