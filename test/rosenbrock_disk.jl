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

## Test dual evaluator

# TODO: init_iterate is really mostly for initializing the type, abusing here to get the right values.
f_and_jac(fx, jac, zeros(n))
iterate = SparseCCSA.init_iterate(; n, m, x0 = zeros(2), jac_prototype = jac, lb=[-Inf, -Inf], ub=[Inf, Inf])    
iterate.fx .= fx
buffers = SparseCCSA.init_buffers(; T=Float64, n)
dual_evaluator = SparseCCSA.DualEvaluator(; iterate, buffers)

gλ = zeros(1)
∇gλ = zeros(m)
λ = ones(1)
dual_evaluator(gλ, ∇gλ, λ)
gλ
∇gλ
dual_evaluator.buffers.δ

## Solve optimization problem

opt = init(f_and_jac, fill(typemin(0.0), 2), fill(typemax(0.0), 2), n, m; x0 = [0.0, 0.0], max_iters = 5,
           max_inner_iters = 5,
          max_dual_iters = 2, max_dual_inner_iters = 5, jac_prototype = zeros(m + 1, n));

step!(opt)