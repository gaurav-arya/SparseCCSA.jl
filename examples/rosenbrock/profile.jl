includet("DefineRosenbrock.jl")
using .DefineRosenbrock
using SparseCCSA

opt = init(f_and_jac, n, m, Float64, zeros(m+1, n);
            lb=fill(-1.0, 2), ub=fill(2.0, 2),
            x0 = [0.5, 0.5], max_iters = 100000000,
            dual_ftol_abs=1e-15, dual_ftol_rel=1e-15
) 

import Profile
@btime f_and_jac($(zeros(2)), $(zeros(m+1, n)), $(zeros(n))) # check rosenbrock f is efficient
@btime step!(opt) # benchmark optimization step
@profview for i in 1:10000 step!(opt) end # profile optimization step



