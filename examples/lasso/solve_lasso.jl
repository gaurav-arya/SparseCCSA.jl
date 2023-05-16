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
uestsp
uest

h.inner_history[1].gx_proposed
h.inner_history[1].fx_proposed

begin
using CairoMakie
f = Figure(resolution=(1200, 800))
ax1 = Axis(f[1,1], xlabel="Iterations", ylabel="log(value)", yscale=log10)
ax2 = Axis(f[1,2], xlabel="Iterations", ylabel="log(value)", yscale=log10)
ax3 = Axis(f[2,1], xlabel="Iterations", ylabel="log(value)", yscale=log10)
ax4 = Axis(f[2,2], xlabel="Iterations", ylabel="number")
ax5 = Axis(f[1,3], xlabel="Iterations", ylabel="value")
# objective
lines!(ax1, (a -> a[1]).(h.fx), label="fx[1] (objective)")
# lines!(ax1, (a -> a[2]).(h.fx), label="fx[2] (constraint 1)")
# lines!(ax1, (a -> a[3]).(h.fx), label="fx[3] (constraint 2)")
# ρ
lines!(ax2, (a -> a[1]).(h.ρ), label="ρ[1]")
lines!(ax2, (a -> a[2]).(h.ρ), label="ρ[2]")
lines!(ax2, (a -> a[p+1]).(h.ρ), label="ρ[p+1]")
# σ
lines!(ax3, (a -> a[1]).(h.σ), label="σ[1]")
lines!(ax3, (a -> a[2]).(h.σ), label="σ[2]")
lines!(ax3, (a -> a[p+1]).(h.σ), label="σ[p+1]")
# dual iterations
lines!(ax4, Base.Fix2(size, 1).(h.inner_history), label="inner iterations")
lines!(ax4, (ih -> sum(i -> i.dual_iters, ih.dual_info)).(h.inner_history), label="total dual iterations")
# u/t values 
lines!(ax5, (a -> a[1]).(h.x), label="u[1]") 
lines!(ax5, (a -> a[p+1]).(h.x), label="t[1]")    
lines!(ax5, (a -> a[2]).(h.x), label="u[2]") 
lines!(ax5, (a -> a[p+2]).(h.x), label="t[2]")    
# legends
axislegend(ax1; position=:rt)
axislegend(ax2; position=:rt)
axislegend(ax3; position=:rt)
axislegend(ax4; position=:rt)
axislegend(ax5; position=:rt)
f
end

begin
(;f_and_jac, jac_prototype) = lasso_epigraph(G, y, α)
fx = zeros(2p+1)
_u = uest#sp#h.x[end][1:p]
f_and_jac(fx, jac_prototype, vcat(_u, abs.(_u)))
fx
jac_prototype
end

norm(uestnl - uest) / norm(uest)
norm(uestsp - uest) / norm(uest)

## OK, time to try NLopt instead

includet("NLoptLassoData.jl")
using .NLoptLassoData
begin
sol = run_once_nlopt(G, y, α, β)
uestnl = sol[2][1:p]
norm(uestnl - uest_qr) / norm(uest_qr)
end

sol[1]

# evaluate objective on QR soln
begin
grad = zeros(2p)
make_obj(G, y, α, β)(vcat(uest_qr, abs.(uest_qr)), grad)
# make_obj(G, y, α, β)(vcat(uestnl, abs.(uestnl)), grad)
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
