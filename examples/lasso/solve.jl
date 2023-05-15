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

(;f_and_jac, jac_prototype) = lasso_epigraph(G, y, α)

u0 = rand(p)
t = 2 * abs.(u0) # start the t's with some slack
u_and_t = vcat(u0, t)

opt = init(f_and_jac, n, m, Float64, jac_prototype;
            lb=fill(-1.0, 2), ub=fill(2.0, 2),
            x0 = [0.5, 0.5], max_iters = iters,
            dual_ftol_abs=1e-15, dual_ftol_rel=1e-15
) 

