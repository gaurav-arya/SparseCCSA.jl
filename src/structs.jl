"""
Cache of all mutable information maintained by CCSAOptimizer.
"""
@kwdef struct CCSACache{T, L}
    ## Information about the current primal cache, which is sufficient to specify the dual problem.
    x::Vector{T} # (n x 1) x 1 cache xᵏ
    fx::Vector{T} # (m+1) x 1 values of objective and constraints
    jac_fx::L # (m+1) x n Jacobian linear operator at x
    ρ::Vector{T} # (m+1) x 1 penality weight
    σ::Vector{T} # n x 1 axes lengths of trust region
    lb::Vector{T} # n x 1 lower bounds on solution
    ub::Vector{T} # n x 1 upper bounds on solution

    ## Buffers used by inner iteration logic.
    Δx_proposed::Vector{T}
    x_proposed::Vector{T}
    gx_proposed::Vector{T} # (m+1) x 1 values of approximate objective and constraints at x_proposed
    fx_proposed::Vector{T} # (m+1) x 1 values of true objective and constraints at x_proposed
    jac_fx_proposed::L # (m+1) x n Jacobian linear operator at x_proposeed

    ## Buffers used in outer iteration logic.
    x_prev::Vector{T} # n x 1 xᵏ⁻¹ 
    x_prevprev::Vector{T} # n x 1 xᵏ⁻²
    fx_prev::Vector{T} # (m+1) x 1 previous objective and constraints

    ## Buffers used in dual evaluation
    a::Vector{T} # n x 1 buffer
    b::Vector{T} # n x 1 buffer
    Δx::Vector{T} # n x 1 buffer
    λ_all::Vector{T} # (m + 1) x 1 buffer
    grad_gλ_all::Vector{T} # (m + 1) x 1 buffer
end

function allocate_cache(; n, m, T, jac_prototype)
    return CCSACache(; x = zeros(T, n), fx = zeros(T, m + 1), jac_fx = copy(jac_prototype),
                   ρ = zeros(T, m + 1),
                   σ = zeros(T, n), lb = zeros(T, n), ub = zeros(T, n),
                   Δx_proposed = zeros(T, n), x_proposed = zeros(T, n), gx_proposed = zeros(T, m + 1),
                   fx_proposed = zeros(T, m + 1), jac_fx_proposed = copy(jac_prototype),
                   x_prev = zeros(T, n), x_prevprev = zeros(T, n), fx_prev = zeros(T, m + 1),
                   a = zeros(T, n), b = zeros(T, n), Δx = zeros(T, n), 
                   λ_all = zeros(T, m + 1), grad_gλ_all = zeros(T, m + 1))
end

"""
Instantiates the cache structure for a dual problem with m constraints.
"""
function allocate_cache_for_dual(; m, T)
    return allocate_cache(; n = m, m = 0, T,
                        jac_prototype = zeros(T, 1, m))
end

"""
Settings for the CCSA optimization, e.g. stopping conditions.
"""
@kwdef mutable struct CCSASettings{T1, T2, T3, T4, T5, T6}
    xtol_rel::T1 = nothing # relative tolerence of solution
    xtol_abs::T2 = nothing # absolute tolerence of solution
    ftol_rel::T3 = nothing # relative tolerence of objective 
    ftol_abs::T4 = nothing # absolute tolerence of objective
    max_iters::T5 = nothing # max number of iterations
    max_inner_iters::T6 = nothing # max number of inner iterations
end
# TODO: ensure at least one stopping condition
# function has_stopping_condition(settings::CCSASettings) 
#     return (settings.xtol_rel > zero(T)) || (settings.xtol_abs > zero(T)) ||
#            (settings.ftol_rel > zero(T)) || (settings.ftol_abs > zero(T)) ||
#            (settings.max_iters < typemax(T))
# end

"""
Statistics maintained by the CCSA optimizer.
"""
@kwdef mutable struct CCSAStats{H}
    outer_iters_done::Int = 0
    inner_iters_done::Int = 0
    inner_iters_cur_done::Int = 0
    history::H = DataFrame()
end
function reset!(stats::CCSAStats)
    stats.outer_iters_done = 0
    stats.inner_iters_done = 0
end

"""
Steppable CCSAOptimizer structure.
"""
@kwdef struct CCSAOptimizer{T, F, L, D, H}
    f_and_jac::F # f(x) = (m+1, (m+1) x n linear operator)
    cache::CCSACache{T, L}
    dual_optimizer::D
    settings::CCSASettings
    stats::CCSAStats{H} = CCSAStats()
end

"""
Solution structure, not yet fleshed out / used.
"""
@kwdef struct Solution{T}
    x::Vector{T}
    RET::Symbol
end