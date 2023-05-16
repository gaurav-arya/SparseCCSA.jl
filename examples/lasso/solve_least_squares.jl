includet("SetupLasso.jl")
using .SetupLasso
using LinearAlgebra

## Initialize problem

begin
n = 8
p = 8
S = 2
(;u, G, y) = setup_lasso(n, p, S)
α = 0.0#1e-2
β = 1e-5#0.0
end

## Solve problem with FISTA

begin
using ImplicitAdjoints
uest, info = genlasso(G, y, α, β, 1000000, 1e-44, L1(p))
norm(uest - u) / norm(u)
end

## Solve problem with QR

uest_qr = (G'G + β * I) \ G'y
norm(uest - uest_qr) / norm(uest_qr)

## Solve problem with CCSA

includet("SparseCCSALassoData.jl")
using .SparseCCSALassoData

begin
h = sparseccsa_lasso_data(G, y, α, β)
ih = h.inner_history[1]
uestsp = h.x[end][1:p]
norm(uestsp - uest) / norm(uest)
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

uestsp
uest

begin
grad = zeros(2p)
NLoptLassoData.make_obj(G, y, α)(vcat(uest, abs.(uest)), grad)
gradnl = zeros(2p)
NLoptLassoData.make_obj(G, y, α)(vcat(uestnl, abs.(uestnl)), gradnl)
end

repr(grad)
repr(gradnl)
repr(uest)
uestnl
grad[2]

uest

NLoptLassoData.make_cons(p, Val(9))(vcat(uestnl, abs.(uestnl)), grad)
repr(grad)

norm(uestsp - uestnl) / norm(uestnl)