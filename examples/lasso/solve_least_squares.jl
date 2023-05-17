includet("SetupLasso.jl")
using .SetupLasso
using LinearAlgebra

## Initialize problem

begin
n = 8
p = 8
S = 2
noise_level = 0.0
(;u, G, y) = setup_lasso(n, p, S, noise_level)
α = 0.0#1e-2
β = 1e-1#0.0
end

## Solve problem with FISTA

includet("FISTASolver.jl")
using .FISTASolver
begin
uest, info = fista(G, y, α, β, 1000000, 1e-44, L1(p))
norm(uest - u) / norm(u)
end

## Solve problem with QR

uest_qr = (G'G + β * I) \ G'y
norm(uest - uest_qr) / norm(uest_qr)

## Solve problem with CCSA

includet("SparseCCSALassoData.jl")
using .SparseCCSALassoData

begin
h, opt = sparseccsa_lasso_data(G, y, α, β; xtol_rel=1e-10, dual_ftol_rel=1e-10)
ih = h.inner_history[1]
uestsp = h.x[end][1:p]
@show norm(uestsp - uest_qr) / norm(uest_qr) # note: we can get better convergence with no noise.
@show opt.stats.inner_iters_done
end

## OK, time to try NLopt instead

includet("NLoptLassoData.jl")
using .NLoptLassoData
begin
sol = run_once_nlopt(G, y, α, β)
uestnl = sol[2][1:p]
norm(uestnl - uest_qr) / norm(uest_qr)
end

# evaluate objective on QR soln
begin
grad = zeros(2p)
@show make_obj(G, y, α, β)(vcat(uest_qr, abs.(uest_qr)), grad)
@show make_obj(G, y, α, β)(vcat(uestnl, abs.(uestnl)), grad)
@show make_obj(G, y, α, β)(vcat(uestsp, abs.(uestsp)), grad)
# gradnl = zeros(2p)
# NLoptLassoData.make_obj(G, y, α)(vcat(uestnl, abs.(uestnl)), gradnl)
end
