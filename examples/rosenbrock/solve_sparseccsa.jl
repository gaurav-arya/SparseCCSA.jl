function sparseccsa_df(iters=1)
    opt = init(f_and_jac, fill(-1.0, 2), fill(2.0, 2), n, m;
               x0 = [0.5, 0.5], max_iters = 5,
               max_inner_iters = 150,
               max_dual_iters = 150, max_dual_inner_iters = 150,
               jac_prototype = zeros(m + 1, n))
    dual_optimizer = opt.dual_optimizer

    # d = DataFrame()
    for i in 1:iters
        step!(opt; verbosity=1)
        # push!(d, (;ρ = copy(opt.iterate.ρ), σ = copy(opt.iterate.σ)))
    end
    return opt.history
end