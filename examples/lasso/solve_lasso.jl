includet("SetupLasso.jl")
using .SetupLasso

## Initialize problem

n = 20
p = 40
(;u, G, y) = setup_lasso(n, p)
α = 1e-1
β = 0.0

## Solve problem with FISTA

using ImplicitAdjoints
uest, info = genlasso(G, y, α, β, 10000, 1e-16, L1(p))

## Solve problem with CCSA

using SparseCCSA
(;f_and_jac, jac_prototype) = lasso_epigraph(G, y, α)

u0 = rand(p)
t0 = 2 * abs.(u0) # start the t's with some slack
u0_and_t0 = vcat(u0, t)

opt = init(f_and_jac, 2p, 2p, Float64, jac_prototype;
            lb=vcat(fill(-Inf, p), zeros(p)), ub=Inf,
            x0 = u0_and_t0, max_iters = 1,
            # max_inner_iters=5,
            # max_dual_iters=5,
            # max_dual_inner_iters=5,
            dual_ftol_abs=1e-7, dual_ftol_rel=1e-7
) 
solve!(opt)
