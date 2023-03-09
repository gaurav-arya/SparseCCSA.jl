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

for i in 1:300
    step!(opt)
end

opt.iterate |> dump