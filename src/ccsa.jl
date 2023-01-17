"""
This structure contains information about the current primal
iterate, which is sufficient to specify the dual problem.
"""
struct PrimalIterate{T,L}
    x::Vector{T} # (n x 1) x 1 primal iterate xᵏ
    fx::Vector{T} # (m+1) x 1 values of objective+constraints of primal
    ∇fx::L # (m+1) x n Jacobian linear operator at x
    ρ::Vector{T} # (m+1) x 1 penality weight of primal
    σ::Vector{T} # n x 1 axes lengths of trust region
    lb::Vector{T} # n x 1 lower bounds on primal solution
    ub::Vector{T} # n x 1 upper bounds on primal solution
    Δx::Vector{T} # (n x 1) x 1 xᵏ⁺¹ - xᵏ
    Δx_last::Vector{T} # n x 1 xᵏ - xᵏ⁻¹
end

"""
Mutable buffers used by the dual optimization algorithm.
"""
struct DualBuffers{T}
    a::Vector{T} # n x 1 buffer
    b::Vector{T} # n x 1 buffer
    δ::Vector{T} # n x 1 buffer
end

"""
A callable structure for evaluating the dual function and its gradient.
"""
struct DualEvaluator{T, L}
    iterate::PrimalIterate{T, L}
    buffer::DualBuffers{T}
end

"""
    (evaluator::DualEvaluator{T})(∇gλ, λ)

Return the dual objective gλ at λ and mutate 
the dual gradient ∇gλ in-place to its new value.
"""
function (evaluator::DualEvaluator{T})(∇gλ, λ) where {T}
    @unpack σ, ρ, x, fx, ∇fx, lb, ub = evaluator.iterate
    @unpack a, b, δ = evaluator.buffer
    λ_all = CatView([one(T)], λ)

    @. a = dot(λ_all, ρ) / (2 * σ ^ 2)
    mul!(b, ∇fx', λ_all)
    @. δ = clamp(-b / (2 * a), -σ, σ)
    @. δ = clamp(δ, lb - x, ub - x)
    gλ = dot(λ_all, fx) + sum(@. a * δ^2 + b * δ)
    mul!(∇gλ, ∇fx, δ)
    @. ∇gλ += fx + sum(abs2, δ / σ) / 2 * ρ
    return gλ
end

"""
The static information (i.e. unchanging during optimization) 
needed to specify the optimization problem to be solved by CCSA.
"""
struct CCSASpecification{T,F}
    lb::Vector{T} # n lower bounds
    ub::Vector{T} # n upper bounds
    f_and_∇f::F # f(x) = (m+1, (m+1) x n linear operator)
    xtol_rel::T # relative tolerence
    xtol_abs::T # absolute tolerence
    ftol_rel::T # relative tolerence
    ftol_abs::T # absolute tolerence
    max_iters::Int # max number of iterations
end

"""
    inner_iterate(opt::CCSAState)

Given optimization state opt, evaluate objective and
Jacobian at current x. These can be used to form the dual problem
objective and its gradient (as functions of the dual soln λ).

...
Find new candidate point.

Check if conservative at new candidate point (objective + constraints)
all underestimated. If so, return.
"""
function inner_iterate(opt::CCSAState{T}) where {T}
    gᵢxᵏ⁺¹ = Vector{T}(undef, opt.m + 1)
    conservative = BitVector(undef, opt.m + 1)
    # arbitrary upper bound on inner iterations
    opt.fx, opt.∇fx = opt.f_and_∇f(opt.x) # expect this line to take up the most time. |Ax - b|^2 + λ |x|₁ <- x is length n, and there are n constraints.
    # normally doing A * x takes a lot longer than n time.
    # TODO: below function should be a callable struct
    opt.dual.f_and_∇f = function (λ) # negative Lagrange dual function and gradient
        gλ, ∇gλ = dual_func!(λ, opt)
        -gλ, -∇gλ # minus signs used to change max problem to min problem
    end
    opt.dual.fx, opt.dual.∇fx = opt.dual.f_and_∇f(opt.dual.x)
    optimize_simple(opt.dual)
    #dual_func!(opt.dual.x , opt)# optimal solution of Lagrange dual problem
    gᵢxᵏ⁺¹ .= opt.fx .+ sum(abs2, opt.Δx ./ opt.σ) / 2 .* opt.ρ
    mul!(gᵢxᵏ⁺¹, opt.∇fx, opt.Δx, true, true)
    fᵢxᵏ⁺¹ = opt.f_and_∇f(opt.x + opt.Δx)[1]
    conservative .= gᵢxᵏ⁺¹ .≥ fᵢxᵏ⁺¹
    if all(conservative)
        return
    end
    opt.ρ[.!conservative] *= 2 # increase ρ until achieving conservative approxmation
    opt.dual.ρ .= one(T) # reinitialize penality weights
    opt.dual.σ .= one(T) # reinitialize radii of trust region
    opt.dual.x .= one(T) # reinitialize starting point of Lagrange multipliers
end

function step!(state::CCSAState, spec::CCSASpecification)
    @unpack f_and_∇f = spec 
    @unpack iterate, extra_info = state


    # TODO: why 10 inner iterations here?
    for i in 1:10
        inner_iterate(opt)
    end
    f = opt.fx[1] # Get current value of objective function
    Δf = opt.f_and_∇f(opt.x+opt.Δx)[1][1] - f # Get change in opt val..? 
                                                # TODO: is x + Δx what we want/
    
    monotonic .= (signbit.(opt.Δx_last).==signbit.(opt.Δx))
    opt.σ[monotonic] *= 2 # double trust region if xⱼ moves monotomically
    opt.σ[.!monotonic] /= 2 # shrink trust region if xⱼ oscillates
    opt.ρ /= 2
    opt.x .+= opt.Δx
    opt.Δx_last .= opt.Δx
    opt.iters += 1
end

struct Solution{T}
    x::Vector{T}
    RET::Symbol
end

function optimize!(state::CCSAState, spec::CCSASpecification{T})
end

function optimize(spec::CCSASpecification{T}, x0::Vector{T}, lb::Vector{T}, ub::Vector{T}) where {T}
    fx, ∇fx = spec.f_and_∇f(x0)
    if any(@view f[2:end] .> 0)
        return ODESolution(x0, :INFEASIBLE)
    end
    n = length(x)
    dual_spec = DualSpecification(x0, fx, ∇fx, ones(T, m+1), ones(T, n), ones(T, n), ones(T, n))
    extra_info = ExtraInfo(zeros(T, n), zeros(T, n))
    state = CCSAState(dual_spec, extra_info)
    for iter in 1:spec.max_iters 
        step!(state, spec)
        # Check termination conditions
        if norm(opt.Δx, Inf) < opt.xtol_abs
            opt.RET = :XTOL_ABS
            return
        end
        if norm(opt.Δx, Inf) / norm(opt.x, Inf)  < opt.xtol_rel
            opt.RET = :XTOL_REL
            return
        end
        if norm(Δf, Inf) < opt.ftol_abs
            opt.RET = :FTOL_REL
            return
        end
        if norm(Δf, Inf) / norm(f, Inf) < opt.ftol_rel
            opt.RET = :FTOL_REL
            return
        end
    end
    return Solution(state.dual_spec.x, :MAX_ITERS)
end


"""
    optimize(opt::CCSAState; callback = nothing)

Optimize the mutable optization state opt. 
We assume that opt.fx and opt.∇fx are prepopulated 
with their values at the initial point x.

The final solution can be found in opt.x, and retcode in opt.RET.
"""
function optimize(opt::CCSAState; callback = nothing)
    if opt.f_and_∇f(opt.x)[1][2:end] <= zero(opt.x)
        nothing
    else
        print("x₀ is not feasible.")
        return 
    end

    # Special case when there are no constraints
    if opt.m == 0
        return optimize_simple(opt)
    end
    # is monotonic or oscillating? TODO: this should be part of state. 
    monotonic = BitVector(undef, opt.n)
    #= 
    Perform outer iterations.
    Summary:

    =#
    while true
        # TODO: why 10 inner iterations here?
        for i in 1:10
            inner_iterate(opt)
        end
        f = opt.fx[1] # Get current value of objective function
        Δf = opt.f_and_∇f(opt.x+opt.Δx)[1][1] - f # Get change in opt val..? 
                                                  # TODO: is x + Δx what we want/
        if opt.iters > opt.max_iters
            opt.RET = :MAX_ITERS
            return
        end
        #= 
        Check for xtol and ftol (abs and rel), all using infinity norm.
        TODO: is infinity norm correct here?
        =#
        if norm(opt.Δx, Inf) < opt.xtol_abs
            opt.RET = :XTOL_ABS
            return
        end
        if norm(opt.Δx, Inf) / norm(opt.x, Inf)  < opt.xtol_rel
            opt.RET = :XTOL_REL
            return
        end
        if norm(Δf, Inf) < opt.ftol_abs
            opt.RET = :FTOL_REL
            return
        end
        if norm(Δf, Inf) / norm(f, Inf) < opt.ftol_rel
            opt.RET = :FTOL_REL
            return
        end
        
        monotonic .= (signbit.(opt.Δx_last).==signbit.(opt.Δx))
        opt.σ[monotonic] *= 2 # double trust region if xⱼ moves monotomically
        opt.σ[.!monotonic] /= 2 # shrink trust region if xⱼ oscillates
        opt.ρ /= 2
        opt.x .+= opt.Δx
        opt.Δx_last .= opt.Δx
        opt.iters += 1
        if callback !== nothing
            callback()
        end
    end
    return opt.iters
end


# optimize problem with no constraint
function optimize_simple(opt::CCSAState{T}) where {T}
    monotonic = BitVector(undef, opt.n)
    while opt.iters < opt.max_iters
        opt.fx, opt.∇fx = opt.f_and_∇f(opt.x)
        opt.a .= opt.ρ[1] / 2 ./ (opt.σ) .^ 2
        opt.b .= opt.∇fx[:]
        for i in 1:10
            @. opt.Δx = clamp(-opt.b / (2 * opt.a), -opt.σ, opt.σ)
            @. opt.Δx = clamp(opt.Δx, opt.lb - opt.x, opt.ub - opt.x)
            opt.gλ = opt.fx[1] + sum(@. opt.a * opt.Δx^2 + opt.b * opt.Δx)
            if opt.gλ ≥ opt.f_and_∇f(opt.x + opt.Δx)[1][1] # check conservative
                break
            end
            opt.ρ *= 2
            opt.a *= 2
        end
        opt.ρ /= 2
        monotonic .= signbit.(opt.Δx_last) .== signbit.(opt.Δx) # signbit avoid multiplication
        opt.σ[monotonic] *= 2 # double trust region if xⱼ moves monotomically
        opt.σ[.!monotonic] /= 2 # shrink trust region if xⱼ oscillates
        opt.x .+= opt.Δx
        opt.Δx_last .= opt.Δx
        if norm(opt.Δx, Inf) < opt.xtol_rel
            return
        end
    end
end

