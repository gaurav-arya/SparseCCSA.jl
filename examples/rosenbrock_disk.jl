# https://www.mathworks.com/help/optim/ug/example-nonlinear-constrained-minimization.html?w.mathworks.com
# This file is work-in-progress

using ForwardDiff
using SparseCCSA

function f(x)
    obj = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
    cons = x[1]^2 + x[2]^2 - 1
    return [obj, cons]
end

function f_and_jac(fx, jac, x)
    fx .= f(x)
    if jac !== nothing
        jac .= ForwardDiff.jacobian(f, x)
    end
    nothing
end

n = 2
m = 1

fx = zeros(m + 1)
jac = zeros(m + 1, n)
f_and_jac(fx, jac, zeros(n))

## Solve optimization problem

begin
    opt = init(f_and_jac, fill(-1.0, 2), fill(1.0, 2), n, m;
               x0 = [0.5, 0.5], max_iters = 5,
               max_inner_iters = 20,
               max_dual_iters = 50, max_dual_inner_iters = 50,
               jac_prototype = zeros(m + 1, n))
    dual_optimizer = opt.dual_optimizer
    dual_iterate = dual_optimizer.iterate
end

for i in 1:1
    step!(opt)
end

opt.iterate |> dump

## Try NLOpt

using NLopt

function obj(x, grad)
    if length(grad) > 0
        grad .= ForwardDiff.gradient(x -> f(x)[1], x)
    end
    return f(x)[1]
end

function cons1(x, grad)
    if length(grad) > 0
        grad .= ForwardDiff.gradient(x -> f(x)[2], x)
    end
    return f(x)[2]
end

function run_once_nlopt()
    nlopt = Opt(:LD_MMA, 2)
    nlopt.lower_bounds = [-1.0, -1.0]
    nlopt.maxeval = 6
    # nlopt.xtol_rel = 1e-4
    nlopt.params["verbosity"] = 2

    nlopt.min_objective = obj 
    inequality_constraint!(nlopt, cons1)

    (minf,minx,ret) = optimize(nlopt, [0.5, 0.5])
    return minf,minx,ret
end

function run_once_mine()
    opt = init(f_and_jac, fill(-1.0, 2), fill(1.0, 2), n, m;
               x0 = [0.5, 0.5], max_iters = 5,
               max_inner_iters = 20,
               max_dual_iters = 50, max_dual_inner_iters = 50,
               jac_prototype = zeros(m + 1, n))
    dual_optimizer = opt.dual_optimizer
    dual_iterate = dual_optimizer.iterate

    step!(opt)
    return opt
end

run_once_nlopt()

run_once_mine();
opt.iterate.fx[1]
opt.iterate.x
opt.iterate.ρ

##

nlopt.numevals


f([0.5, 0.5])[1]
f([0.603, 0.397])[1]
f(opt.iterate.x)[1]

