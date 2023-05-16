module SparseCCSALassoData
__revise_mode__ = :eval

include("SetupLasso.jl")
using .SetupLasso
using SparseCCSA

function sparseccsa_lasso_data(G, y, α)
    n, p = size(G)
    
    (;f_and_jac, jac_prototype) = lasso_epigraph(G, y, α)
    u0 = zeros(p)
    t0 = 2 * abs.(u0) # start the t's with some slack
    u0_and_t0 = vcat(u0, t0)

    opt = init(f_and_jac, 2p, 2p, Float64, jac_prototype;
                lb=vcat(fill(-Inf, p), zeros(p)), ub=Inf,
                x0 = u0_and_t0, 
                max_iters = 10000,
                dual_xtol_rel=1e-8,
    ) 
    sol = solve!(opt; verbosity=Val(2))
    return opt.stats.history
end

export sparseccsa_lasso_data
end