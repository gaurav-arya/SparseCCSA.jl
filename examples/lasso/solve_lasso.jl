includet("SetupLasso.jl")
using .SetupLasso
using LinearAlgebra

## Initialize problem

begin
n = 4
p = 7
S = 2
(;u, G, y) = setup_lasso(n, p, S)
α = 1e-2
β = 0.0
end

## Solve problem with FISTA

begin
using ImplicitAdjoints
uest, info = genlasso(G, y, α, β, 10000, 1e-16, L1(p))
norm(uest - u) / norm(u)
end

## Solve problem with CCSA

includet("SparseCCSALassoData.jl")
using .SparseCCSALassoData
using CairoMakie

begin
h = sparseccsa_lasso_data(G, y, α)
ih = h.inner_history[1]
uestsp = h.x[end][1:p]
sum(uestsp)
end

begin
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
u = uestsp#h.x[end][1:p]
f_and_jac(fx, jac_prototype, vcat(u, abs.(u)))
fx
jac_prototype
end

h.fx[64][1]
lines((a -> a[7]).(h.x))


h.inner_history[46].dual_info[2].dual_history.fx
h.fx[end-1]
h.σ[end-1]
h.ρ[end-2]
h.inner_history[end-4].ρ
h.inner_history[end].fx_proposed
h.inner_history[end].gx_proposed

norm(uestsp - uest) / norm(uest)
norm(uestsp - u) / norm(u)

## OK, time to try NLopt instead

function obj(x, grad)
    f_and
    if length(grad) > 0
        grad .= ForwardDiff.gradient(x -> f(x)[1], x)
    end
    return f(x)[1]
end

function consi(x, grad, i)
    iv = SparseCCSA._unwrap_val(i)
    if length(grad) > 0
        grad .= ForwardDiff.gradient(x -> f(x)[iv], x)
    end
    return f(x)[2]
end

function run_once_nlopt(evals)
    nlopt = Opt(:LD_CCSAQ, 2)
    nlopt.lower_bounds = [-1.0, -1.0]
    nlopt.upper_bounds = [2.0, 2.0]
    nlopt.maxeval = evals 
    nlopt.xtol_rel = 0.0
    nlopt.xtol_abs = 0.0
    nlopt.params["verbosity"] = 2
    nlopt.params["max_inner_iters"] = 1

    nlopt.min_objective = obj 
    inequality_constraint!(nlopt, cons1)

    (minf,minx,ret) = optimize(nlopt, [0.5, 0.5])
    return minf,minx,ret
end