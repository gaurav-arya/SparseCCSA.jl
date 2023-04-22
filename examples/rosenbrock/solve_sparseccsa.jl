opt = init(f_and_jac, fill(-1.0, 2), fill(1.0, 2), n, m;
            x0 = [0.5, 0.5], max_iters = 5,
            max_inner_iters = 20,
            max_dual_iters = 50, max_dual_inner_iters = 50,
            jac_prototype = zeros(m + 1, n))

function run_once_mine(iters=1)
    opt = init(f_and_jac, fill(-1.0, 2), fill(1.0, 2), n, m;
               x0 = [0.5, 0.5], max_iters = 5,
               max_inner_iters = 20,
               max_dual_iters = 50, max_dual_inner_iters = 50,
               jac_prototype = zeros(m + 1, n))
    dual_optimizer = opt.dual_optimizer
    dual_iterate = dual_optimizer.iterate

    for i in 1:iters
        step!(opt; verbose=true)
    end
    return opt
end
opt.iterate |> dump