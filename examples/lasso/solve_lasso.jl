includet("SetupLasso.jl")
using .SetupLasso
using LinearAlgebra

## Initialize problem

n = 2
p = 4
S = 1
(;u, G, y) = setup_lasso(n, p, S)
α = 1e-1
β = 0.0

## Solve problem with FISTA

using ImplicitAdjoints
uest, info = genlasso(G, y, α, β, 10000, 1e-16, L1(p))
norm(uest - u) / norm(u)

## Solve problem with CCSA

using SparseCCSA
(;f_and_jac, jac_prototype) = lasso_epigraph(G, y, α)

u0 = rand(p)
t0 = 2 * abs.(u0) # start the t's with some slack
u0_and_t0 = vcat(u0, t0)

jac_prototype
f_and_jac(zeros(2p+1), jac_prototype, u0_and_t0)

opt = init(f_and_jac, 2p, 2p, Float64, jac_prototype;
            lb=vcat(fill(-Inf, p), zeros(p)), ub=Inf,
            x0 = u0_and_t0, 
            max_iters = 1,
            max_inner_iters=100,
            max_dual_iters=100,
            max_dual_inner_iters=100,
            dual_ftol_abs=1e-7, dual_ftol_rel=1e-7
) 
solve!(opt; verbosity=Val(2))

begin
h = opt.stats.history
ih = h.inner_history[1]
end
