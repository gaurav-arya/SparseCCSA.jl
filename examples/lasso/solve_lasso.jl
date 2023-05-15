includet("SetupLasso.jl")
using .SetupLasso
using LinearAlgebra

## Initialize problem

begin
n = 10
p = 10
S = 1
(;u, G, y) = setup_lasso(n, p, S)
α = 1e-6
β = 0.0
end

## Solve problem with FISTA

using ImplicitAdjoints
uest, info = genlasso(G, y, α, β, 10000, 1e-16, L1(p))
norm(uest - u) / norm(u)
uest
u

## Solve problem with CCSA

using SparseCCSA

begin
(;f_and_jac, jac_prototype) = lasso_epigraph(G, y, α)
u0 = rand(p)
t0 = 2 * abs.(u0) # start the t's with some slack
u0_and_t0 = vcat(u0, t0)
end

begin
opt = init(f_and_jac, 2p, 2p, Float64, jac_prototype;
            lb=vcat(fill(-Inf, p), zeros(p)), ub=Inf,
            x0 = u0_and_t0, 
            max_iters = 1000,
            dual_ftol_abs=1e-10, dual_ftol_rel=1e-10
) 
sol = solve!(opt; verbosity=Val(2))
end

begin
h = opt.stats.history
ih = h.inner_history[end]
end

sol.x
uest
u
norm(sol.x[1:p] - uest) / norm(uest)
norm(sol.x[1:p] - u) / norm(u)