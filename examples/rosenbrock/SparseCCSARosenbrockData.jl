module SparseCCSARosenbrockData
__revise_mode__ = :eval

include("DefineRosenbrock.jl")
using .DefineRosenbrock
using SparseCCSA

function sparseccsa_dataframe(iters=1)
    opt = init(f_and_jac, n, m, Float64, zeros(m+1, n);
               lb=fill(-1.0, 2), ub=fill(2.0, 2),
               x0 = [0.5, 0.5], max_iters = iters, xtol_rel=1e-8,
               dual_ftol_abs=0.0, dual_ftol_rel=1e-15
    ) 

    solve!(opt; verbosity=Val(2))
    return opt.stats.history
end

export sparseccsa_dataframe

end