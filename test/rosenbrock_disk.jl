# https://www.mathworks.com/help/optim/ug/example-nonlinear-constrained-minimization.html?w.mathworks.com

using ForwardDiff
using SparseCCSA

function f(x)
    obj = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
    cons = x[1]^2 + x[2]^2 - 1
    return [obj, cons]
end

function f_and_jac(fx, jac, x)
    fx .= f(x)
    jac .= ForwardDiff.jacobian(f, x)
    nothing
end

n = 2
m = 1

fx = zeros(m + 1)
jac = zeros(m+1, n)
f_and_jac(fx, jac, zeros(n))
fx
jac

## Solve optimization problem

opt = init(f_and_jac, fill(typemin(0.0), 2), fill(typemax(0.0), 2), n, m; x0 = [0.5, 0.5], max_iters = 5,
           max_inner_iters = 10,
           max_dual_iters = 20, max_dual_inner_iters = 3, jac_prototype = zeros(m + 1, n));

dual_optimizer = opt.dual_optimizer
dual_evaluator = dual_optimizer.f_and_jac

function evaluate_dual(λ1)
    dual_evaluator(dual_optimizer.iterate.fx, dual_optimizer.iterate.jac_fx, [λ1])
    return (dual_optimizer.iterate.fx, dual_evaluator.buffers.Δx)
end
evaluate_dual(29) # minimum!

using GLMakie
lines(1:50, [evaluate_dual(i)[1][1] for i in 1:50])

## Expected minimum = 29. Can we get there? Yes!

step!(dual_optimizer)