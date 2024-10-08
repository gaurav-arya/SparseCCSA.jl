using Optimization, SparseCCSA, ForwardDiff

x0 = zeros(2)
rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

cons = (res, x, p) -> (res[1] = x[1]^2 + x[2]^2 - 1.0; return nothing)
optf = OptimizationFunction(rosenbrock, AutoSparseForwardDiff(), cons = cons)

prob = OptimizationProblem(
    optf, x0, lb = [-1.0, -1.0], ub = [1.0, 1.0], lcons = [0.0], ucons = [0.0])
opt = solve(prob, SparseCCSA.CCSA())
