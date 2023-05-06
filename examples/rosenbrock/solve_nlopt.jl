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
    nlopt.upper_bounds = [2.0, 2.0]
    nlopt.maxeval = evals 
    nlopt.xtol_rel = 0.0
    nlopt.xtol_abs = 0.0
    nlopt.params["verbosity"] = 2
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
        # skip ignored lines
        ln = readline(buffer)
        if (startswith(ln, "j=") || startswith(ln, "dx =")) || startswith(ln, "u =") || startswith(ln, "v =") || startswith(ln, "dfdx") || startswith(ln, "dfcdx") || startswith(ln, "y:")
           return safe_scanf(buffer, fmt, args...)
        end
        seek(buffer, pos)
        return nothing
    end
    return out
end

buffer = IOBuffer(read(open("nlopt_out.txt"), String)) # easier to copy
readline(buffer)

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
        CCSA xcur[0]=%f
        CCSA xcur[1]=%f
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
                    CCSA sigma[1] -> %f
    """
    infeasible_point_fmt = Scanf.format"""
    CCSA - using infeasible point%s
    """

    buffer = IOBuffer(read(open("nlopt_out.txt"), String)) # easier to copy
    d = DataFrame()
    while true 
        # read one inner iteration
        inner_history = DataFrame()
        inner_iter = 0
        done = false
        while true
            if (out = safe_scanf(buffer, inner_iter_fmt, Int64, (Float64 for i in 1:5)...)) !== nothing
                dual_iters, dual_obj, dual_opt, dual_grad, _x_proposed... = out
                x_proposed = collect(_x_proposed)
            else
                done = true
            end
            if (out = safe_scanf(buffer, inner_iter2_fmt, (Float64 for i in 1:2)...)) !== nothing
                ρ = collect(out)
                do_break = false
            else
                ρ = [NaN, NaN]
                do_break = true
            end
            push!(inner_history, (;dual_iters, dual_obj, dual_opt, dual_grad, ρ, x_proposed))
            do_break && break
        end
        done && break
        safe_scanf(buffer, infeasible_point_fmt, String) # skip infeasible point log in hacky way
        out = safe_scanf(buffer, outer_iter_fmt, (Float64 for i in 1:2)...) 
        if out === nothing
            break
        end
        ρ = collect(out)
        if (out = safe_scanf(buffer, outer_iter_sigma_fmt, (Float64 for i in 1:2)...)) !== nothing
            σ = collect(out)
        else
            σ = [NaN]
        end
        push!(d, (;ρ, σ, inner_history))
    end
    if countlines(copy(buffer)) != 0
        @show countlines(copy(buffer))
        write("debug.txt", read(copy(buffer), String))
    end
    return d
end
