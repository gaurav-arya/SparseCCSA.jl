module SparseCCSAData

include("DefineRosenbrock.jl")
using .DefineRosenbrock
using SparseCCSA

function sparseccsa_dataframe(iters=1)
    opt = init(f_and_jac, n, m, Float64, zeros(m+1, n);
               lb=fill(-1.0, 2), ub=fill(2.0, 2),
               x0 = [0.5, 0.5], max_iters = iters,
               dual_ftol_abs=1e-15, dual_ftol_rel=1e-15
    ) 

    solve!(opt; verbosity=2)
    return opt.stats.history
end

export sparseccsa_dataframe

end