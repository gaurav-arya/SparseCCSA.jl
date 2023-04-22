using NLopt

function obj(x, grad)
    if length(grad) > 0
        grad .= ForwardDiff.gradient(x -> f(x)[1], x)
    end
    return f(x)[1]
end

function cons1(x, grad)
    if length(grad) > 0
        grad .= ForwardDiff.gradient(x -> f(x)[2], x)
    end
    return f(x)[2]
end

function run_once_nlopt(iters=1, evals=nothing)
    evals = (evals === nothing) ? special_iters[iters] + 1 : evals
    nlopt = Opt(:LD_CCSAQ, 2)
    nlopt.lower_bounds = [-1.0, -1.0]
    nlopt.upper_bounds = [1.0, 1.0]
    nlopt.maxeval = evals 
    # nlopt.xtol_rel = 1e-4
    nlopt.params["verbosity"] = 1
    nlopt.params["max_inner_iters"] = 1

    nlopt.min_objective = obj 
    inequality_constraint!(nlopt, cons1)

    (minf,minx,ret) = optimize(nlopt, [0.5, 0.5])
    return minf,minx,ret
end

## NLOpt output processing

using DataFrames
using Scanf

function safe_scanf(buffer, fmt, args...)
    pos = position(buffer)
    n, out... = scanf(buffer, fmt, args...)
    if n != length(args)
        seek(buffer, pos)
        return nothing
    end
    return out
end

open("nlopt_out.txt", "w") do io
    redirect_stdout(io) do
        run_once_nlopt(nothing, 20)
        Base.Libc.flush_cstdio()
    end
end
out_str = read(open("nlopt_out.txt"), String)
lines = split(out_str, "\n")

nlopt_res = DataFrame()

buffer = IOBuffer(out_str)

inner_iter_fmt = Scanf.format"""
CCSA dual converged in %d iters to g=%f:
    CCSA y[0]=%f, gc[0]=%f
"""
inner_iter2_fmt = Scanf.format"""
CCSA inner iteration: rho -> %f 
                CCSA rhoc[0] -> %f
"""
outer_iter_fmt = Scanf.format"""
CCSA outer iteration: rho -> %f
                 CCSA rhoc[0] -> %f
"""
outer_iter_sigma_fmt = Scanf.format"""
                 CCSA sigma[0] -> %f
"""
buffer = IOBuffer(out_str)
safe_scanf(buffer, inner_iter, Int64, (Float64 for i in 1:3)...) 
safe_scanf(buffer, inner_iter2, (Float64 for i in 1:2)...) 
safe_scanf(buffer, outer_iter, (Float64 for i in 1:2)...) 
safe_scanf(buffer, outer_iter_sigma, Float64) 

open("out.txt", "w") do f
    write(f, read(copy(buffer), String)) 
end

buffer = IOBuffer(out_str)
while true 
    # read one inner iteration
    d = DataFrame()
    inner_iter = 0
    while true
        dual_iters, dual_obj = safe_scanf(buffer, inner_iter_fmt, Int64, (Float64 for i in 1:3)...) 
        if (out = safe_scanf(buffer, inner_iter2_fmt, (Float64 for i in 1:2)...)) !== nothing
            rho, rhoc = out
        else
            rho, rhoc = NaN, NaN
            break
        end
        push!(d, (;dual_iters, dual_obj, rho, rhoc))
    end
    break
    # push!(nlopt_res, Dict(:iter => iter, :fx => NaN, :x => NaN, :ρ => NaN, :σ => NaN))
end

@scanf
