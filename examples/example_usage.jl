using SparseCCSA
using StaticArrays

# Simple problem:
# maximize x1^2 * x_2
# subject to x_1 <= 3
# and x_1 <= -2
# and x_2 <= 4

function f_and_∇f(fx, ∇fx, x)
    # TODO: remove allocations in below so can test allocation-free'ness of algorithm.
    fx .= [x[1]^2*x[2], x[1] - 3, x[1] + 4, x[2] - 4]
    ∇fx .= [2*x[1] 1; 1 0; 1 0; 0 1]
    return nothing
end

n = 2
m = 3

f_and_∇f(zeros(m+1), zeros(m+1, n), zeros(n))

# Form optimizer
opt = init(f_and_∇f, [0.0, 0.0], [100.0, 100.0], n, m; x0=zeros(n), max_iters=5, max_inner_iters=5, 
            max_dual_iters=5, max_dual_inner_iters=5, ∇fx_prototype = zeros(m+1, n))

st = CCSAState(2, # n
               3, # m
               f_and_grad,
               [-5.0, -5.0], # x₀
               ρ = ones(m + 1), # ρ
               σ = ones(n), # σ
               lb = [0.0, 0.0], # lb
               ub = [100.0, 100.0], # ub
               # the stuff below shouldn't matter
               # later we will ensure that the user does not have to create these, but don't worry about it for now.
               max_iters = 10)

optimize(st) # get answer
