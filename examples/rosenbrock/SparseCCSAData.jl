module SparseCCSAData

include("DefineRosenbrock.jl")
using .DefineRosenbrock
using SparseCCSA

function sparseccsa_dataframe(iters=1)
    opt = init(f_and_jac, n, m, Float64, zeros(m+1, n);
               lb=fill(-1.0, 2), ub=fill(2.0, 2),
               x0 = [0.5, 0.5], max_iters = 400,
               max_inner_iters = 150,
               max_dual_iters = 150, max_dual_inner_iters = 150)

    solve!(opt)
    return opt.history
end

export sparseccsa_dataframe

end