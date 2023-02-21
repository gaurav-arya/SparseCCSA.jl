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
fx
jac

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

## Dual optimization (can we stop σ and ρ blowing up?)
step!(dual_optimizer)
dual_optimizer.iterate |> dump 

## Back to regular optimization

bad_x = [0.627345, 0.390871]
opt.iterate.x .= [0.2, 0.2] #
opt.iterate.ρ .= 1.0
opt.iterate.σ .= 1.0
for i in 1:10
step!(opt)
end

## Checking dual opt

begin
    dual_iterate.x .= [31.75]
    dual_iterate.ρ .= [0.25]
    dual_iterate.σ .= [1.0]
    dual_iterate.Δx .= [15.75]
    dual_optimizer.f_and_jac(dual_iterate.fx, dual_iterate.jac_fx, dual_iterate.x)
end
dual_iterate.fx
dual_iterate.jac_fx
-dual_iterate.jac_fx[1] * dual_iterate.σ[1]^2 / (2 * dual_iterate.ρ[1])

Δx_proposed = [0.0]
SparseCCSA.propose_Δx!(Δx_proposed, dual_optimizer)
Δx_proposed

step!(dual_optimizer)

dual_iterate |> dump
dual_optimizer.f_and_jac.buffers.Δx

function evaluate_dual(λ1)
    dual_evaluator(dual_iterate.fx, dual_iterate.jac_fx, [λ1])
    return (dual_iterate.fx, dual_evaluator.buffers.Δx)
end
evaluate_dual(29) # minimum!

using GLMakie
lines(1:50, [evaluate_dual(i)[1][1] for i in 1:50])

## Expected minimum = 29. Can we get there? Yes!

## Try NLOpt

using NLopt


