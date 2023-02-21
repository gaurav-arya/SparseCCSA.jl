using NLopt

function myfunc(x, grad)
    if length(grad) > 0
        grad[1] = 0
        grad[2] = 0.5/sqrt(x[2])
    end
    return sqrt(x[2])
end

function myconstraint(x, grad, a, b)
    if length(grad) > 0
        grad[1] = 3a * (a*x[1] + b)^2
        grad[2] = -1
    end
    (a*x[1] + b)^3 - x[2]
end

opt = Opt(:LD_MMA, 2)
opt.lower_bounds = [-Inf, 0.]
opt.xtol_rel = 1e-4

opt.min_objective = myfunc
inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 0.0)
inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 0.0)

(minf,minx,ret) = optimize(opt, [1.234, 5.678])
numevals = opt.numevals # the number of function evaluations
println("got $minf at $minx after $numevals iterations (returned $ret)")

## Compare to ours

using SparseCCSA

function f_and_jac(fx, jac_fx, x)
    @show jac_fx
    fx[1] = myfunc(x, (jac_fx !== nothing) ? (@view jac_fx[1, :]) : [])
    fx[2] = myconstraint(x, (jac_fx !== nothing) ? (@view jac_fx[2, :]) : [], 2, 0)
    fx[3] = myconstraint(x, (jac_fx !== nothing) ? (@view jac_fx[1, :]) : [], 3, 0)
    nothing
end

n = 2
m = 2

opt = init(f_and_jac, [-Inf, 0.], [Inf, Inf], n, m;
            x0 = zeros(2), max_iters = 20,
            max_inner_iters = 20,
            max_dual_iters = 50, max_dual_inner_iters = 50,
            jac_prototype = zeros(m + 1, n))

step!(opt)