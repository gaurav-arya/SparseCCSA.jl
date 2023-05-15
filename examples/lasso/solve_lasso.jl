includet("SetupLasso.jl")
using .SetupLasso
using LinearAlgebra

## Initialize problem

begin
n = 4
p = 4
S = 2
(;u, G, y) = setup_lasso(n, p, S)
α = 1e-3
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
end

h[1, :].ρ
h.inner_history[1].ρ[1]

h.inner_history[1].dual_info[1]

h.inner_history[1].dual_history[1] # this is a problem!

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