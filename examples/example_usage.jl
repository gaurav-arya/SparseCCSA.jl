using SparseCCSA
using StaticArrays

## Formulate simple problem 

# maximize x_1^2 * x_2
# and x_1 <= 3
# and x_2 <= 4
function f_and_jac(fx, jac, x)
    # TODO: remove allocations in below so can test allocation-free'ness of algorithm.
    fx .= [x[1]^2 * x[2], x[1] - 3, x[2] - 4]
    jac .= [2*x[1] 1; 1 0; 0 1]
    return nothing
end

n = 2
m = 2

fx = zeros(m + 1)
jac = zeros(m+1, n)
f_and_jac(fx, jac, zeros(n))
fx
jac

## Solve with SparseCCSA

# Form optimizer
opt = init(f_and_jac, [0.0, 0.0], [100.0, 100.0], n, m; x0 = [1.0, 1.0], max_iters = 5,
           max_inner_iters = 5,
          max_dual_iters = 2, max_dual_inner_iters = 5, jac_prototype = zeros(m + 1, n));
try step!(opt) catch e end
step!(opt)

opt.iterate.x
