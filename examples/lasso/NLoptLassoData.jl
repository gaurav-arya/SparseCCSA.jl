module NLoptLassoData
__revise_mode__ = :eval

include("SetupLasso.jl")
using .SetupLasso
using NLopt

function make_obj(G, y, α) 
    n, p = size(G)
    return (u_and_t, grad) -> begin
        u = @view u_and_t[1:p]
        t = @view u_and_t[(p + 1):(2p)]
        if length(grad) > 0
            grad .= vcat(2 * G' * (G * u - y), fill(α, p))
        end
        return sum((G * u - y) .^ 2) + α * sum(t)
    end
end

_unwrap_val(::Val{x}) where x = x
function make_cons(p, iv)
    i = _unwrap_val(iv)
    return (u_and_t, grad) -> begin
        u = @view u_and_t[1:p]
        t = @view u_and_t[(p + 1):(2p)]
        if length(grad) > 0
            grad .= 0
            if i <= p 
                grad[i] = 1
                grad[i + p] = -1
            else
                grad[i - p] = -1
                grad[i] = -1
            end
        end
        return (i <= p) ? u[i] - t[i] : -u[i - p] - t[i - p]
    end
end

function run_once_nlopt(G, y, α)
    n, p = size(G)

    nlopt = Opt(:LD_CCSAQ, 2p)
    nlopt.lower_bounds = vcat(fill(-Inf, p), zeros(p)) 
    nlopt.upper_bounds = fill(Inf, 2p)
    nlopt.maxeval = 10000 
    nlopt.xtol_rel = 1e-6
    nlopt.xtol_abs = 0.0
    nlopt.params["dual_ftol_rel"] = 0.0 # this is crucial! the default 1e-14... not good enough
    nlopt.params["dual_xtol_rel"] = 1e-12
    nlopt.params["verbosity"] = 0

    nlopt.min_objective = make_obj(G, y, α)
    for i in 1:2p
        inequality_constraint!(nlopt, make_cons(p, Val(i)))
    end

    u0 = zeros(p)
    t0 = abs.(u0) # start the t's with some slack
    u0_and_t0 = vcat(u0, t0)

    (minf,minx,ret) = optimize(nlopt, u0_and_t0)
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

"""
    Create DataFrame with NLopt optimization history.
    Requires use of custom nlopt binary: build from https://github.com/gaurav-arya/nlopt/tree/ag-debug,
    move shared object file to this directory, and set with set_binary.jl.
"""
function nlopt_lasso_data(evals) 
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
        push!(d, (;ρ, σ, inner_history, x=inner_history.x_proposed[end]))
    end
    if countlines(copy(buffer)) != 0
        @show countlines(copy(buffer))
        write("debug.txt", read(copy(buffer), String))
    end
    return d
end

export run_once_nlopt, nlopt_lasso_data

end