using SparseCCSA
using StaticArrays

## Formulate simple problem 

# maximize x_1^2 * x_2
# and x_1 <= 3
# and x_2 <= 4
function f_and_∇f(fx, ∇fx, x)
    # TODO: remove allocations in below so can test allocation-free'ness of algorithm.
    fx .= [x[1]^2 * x[2], x[1] - 3, x[2] - 4]
    ∇fx .= [2*x[1] 1; 1 0; 0 1]
    return nothing
end

n = 2
m = 2

fx = zeros(m + 1)
∇fx = zeros(m+1, n)
f_and_∇f(fx, ∇fx, zeros(n))
fx
∇fx

## Solve with SparseCCSA

# Form optimizer
begin
opt = init(f_and_∇f, [0.0, 0.0], [100.0, 100.0], n, m; x0 = [1.0, 1.0], max_iters = 5,
           max_inner_iters = 5,
          max_dual_iters = 2, max_dual_inner_iters = 5, ∇fx_prototype = zeros(m + 1, n));
try step!(opt) catch e end
end

opt.iterate.x
