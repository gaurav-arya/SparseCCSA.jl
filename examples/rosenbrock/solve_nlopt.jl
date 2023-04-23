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

function run_once_nlopt(evals)
    nlopt = Opt(:LD_CCSAQ, 2)
    nlopt.lower_bounds = [-1.0, -1.0]
    nlopt.upper_bounds = [1.0, 1.0]
    nlopt.maxeval = evals 
    nlopt.xtol_rel = 0.0
    nlopt.xtol_abs = 0.0
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

function nlopt_df(evals) 
    open("nlopt_out.txt", "w") do io
        redirect_stdout(io) do
            run_once_nlopt(evals)
            Base.Libc.flush_cstdio()
        end
    end

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
    infeasible_point_fmt = Scanf.format"""
    CCSA - using infeasible point%s
    """

    buffer = IOBuffer(read(open("nlopt_out.txt"), String)) # easier to copy
    d = DataFrame()
    while true 
        # read one inner iteration
        d_inner = DataFrame()
        inner_iter = 0
        done = false
        while true
            if (out = safe_scanf(buffer, inner_iter_fmt, Int64, (Float64 for i in 1:3)...)) !== nothing
                dual_iters, dual_obj = out
            else
                done = true
            end
            if (out = safe_scanf(buffer, inner_iter2_fmt, (Float64 for i in 1:2)...)) !== nothing
                ρ = collect(out)
            else
                ρ = [NaN, NaN]
                break
            end
            push!(d_inner, (;dual_iters, dual_obj, ρ))
        end
        done && break
        safe_scanf(buffer, infeasible_point_fmt, String) # skip infeasible point log in hacky way
        out = safe_scanf(buffer, outer_iter_fmt, (Float64 for i in 1:2)...) 
        if out === nothing
            break
        end
        ρ = collect(out)
        if (out = safe_scanf(buffer, outer_iter_sigma_fmt, Float64)) !== nothing
            σ = collect(out)
        else
            σ = [NaN]
        end
        push!(d, (;d_inner, ρ, σ))
    end
    if countlines(copy(buffer)) != 0
        @show countlines(copy(buffer))
        write("debug.txt", read(copy(buffer), String))
    end
    return d
end

nlopt_df(500)[1:9, :]
