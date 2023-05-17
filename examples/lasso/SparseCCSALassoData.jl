module SparseCCSALassoData
__revise_mode__ = :eval

include("SetupLasso.jl")
using .SetupLasso
using SparseCCSA

function sparseccsa_lasso_data(G, y, α, β; settings...)
    n, p = size(G)
    
    (;f_and_jac, jac_prototype) = lasso_epigraph(G, y, α, β)
    u0 = zeros(p)
    t0 = 2 * abs.(u0) # start the t's with some slack
    u0_and_t0 = vcat(u0, t0)

    opt = init(f_and_jac, 2p, 2p, Float64, jac_prototype;
                lb=vcat(fill(-Inf, p), zeros(p)), ub=Inf,
                x0 = u0_and_t0, 
                settings...
                # xtol_rel=1e-5,
                # max_iters = 1000,
                # max_inner_iters=20,
                # max_dual_iters=20,
                # max_dual_inner_iters=20,
                # dual_xtol_rel=1e-6,
                # dual_ftol_rel=1e-12,
    ) 
    sol = solve!(opt; verbosity=Val(2))
    return opt.stats.history, opt
end

export sparseccsa_lasso_data
end