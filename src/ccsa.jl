"""
An unambiguous specification of the dual problem.
"""
struct DualSpecification{T,L}
    m::Int # number of variables
    x::Vector{T} # n x 1 primal iterate 
    fx::Vector{T} # (m+1) x 1 values of objective+constraints of primal
    ∇fx::L # (m+1) x n Jacobian linear operator at x
    ρ::Vector{T} # (m+1) x 1 penality weight of primal
    σ::Vector{T} # n x 1 axes lengths of trust region
    lb::Vector{T} # n x 1 lower bounds on primal solution
    ub::Vector{T} # n x 1 upper bounds on primal solution
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
    spec::DualSpecification{T, L}
    buffers::DualBuffers{T} 
end

"""
    (evaluator::DualEvaluator{T})(∇gλ, λ)

Return the dual objective gλ at λ and mutate 
the dual gradient ∇gλ in-place to its new value.
"""
function (evaluator::DualEvaluator{T})(∇gλ, λ) where {T}
    @unpack σ, ρ, fx, ∇fx, lb, ub = evaluator.spec 
    @unpack a, b, δ = evaluator.buffers
    λ_all = CatView([one(T)], λ)

    @. a = dot(λ_all, ρ) / (2 * σ ^ 2)
    mul!(b, ∇fx', λ_all)
    @. δ = clamp(-b / (2 * a), -σ, σ)
    @. δ = clamp(δ, lb - p.x, ub - x)
    gλ = dot(λ_all, fx) + sum(@. a * δ^2 + b * δ)
    mul!(∇gλ, ∇fx, δ)
    @. ∇gλ += fx + sum(abs2, δ / σ) / 2 * ρ
    return gλ
end


opt.dual.f_and_∇f = function (λ) # negative Lagrange dual function and gradient
    gλ, ∇gλ = dual_func!(λ, opt)
    -gλ, -∇gλ # minus signs used to change max problem to min problem
end

mutable struct CCSAState{T <: AbstractFloat, F, L}
    n::Int # number of variables > 0
    m::Int # number of inequality constraints ≥ 0
    lb::Vector{T} # n lower bounds
    ub::Vector{T} # n upper bounds
    f_and_∇f::F # f(x) = (m+1, (m+1) x n linear operator)
    ρ::Vector{T} # m + 1 penality weight
    σ::Vector{T} # n radius of trust region
    x::Vector{T} # current best feasible point
    xtol_rel::T # relative tolerence
    xtol_abs::T # absolute tolerence
    ftol_rel::T # relative tolerence
    ftol_abs::T # absolute tolerence
    max_iters::Int # max number of iterations
    iters::Int # interation count

    fx::Vector{T} # (m+1) x 1 function values at x
    ∇fx::L # (m+1) x n linear operator of Jacobian at x
    a::Vector{T} # n
    b::Vector{T} # n
    Δx::Vector{T} # n xᵏ⁺¹ - xᵏ
    Δx_last::Vector{T} # n xᵏ - xᵏ⁻¹
    gλ::T # Lagrange dual function value
    ∇gλ::Vector{T} # m Lagrange dual function gradient
    RET::Symbol
    fx_last::Vector{T}
    dual::CCSAState{T} # Lagrange dual problem if the primal problem has constraints
    

    function CCSAState(n::Int, # number of variables
                       m::Int, # number of inequality constraints
                       f_and_∇f::F,
                       x₀::Vector{T}; # initial feasible point
                       # parameters below are optional
                       ρ::Vector{T} = ones(T, m + 1), # (m + 1) penality weight ρ
                       σ::Vector{T} = ones(T, n), # n radius of trust region σ
                       lb::Vector{T} = fill(typemin(T), n), # lower bounds, default -Inf
                       ub::Vector{T} = fill(typemax(T), n), # upper bounds, default Inf
                       xtol_rel::T = T(1e-5), # relative tolerence
                       xtol_abs::T = T(1e-5), # relative tolerence
                       ftol_rel::T = T(1e-5), # relative tolerence
                       ftol_abs::T = T(1e-5), # relative tolerence
                       max_iters::Int = typemax(Int64)
                       ) where {F, T <: AbstractFloat}
        fx, ∇fx = f_and_∇f(x₀)
        opt = new{T,F,typeof(∇fx)}(n,
                     m,
                     lb,
                     ub,
                     f_and_∇f,
                     ρ,
                     σ,
                     x₀,
                     xtol_rel,
                     xtol_abs,
                     ftol_rel,
                     ftol_abs,
                     max_iters,
                     zero(max_iters),
                     fx,
                     ∇fx,
                     Vector{T}(undef, n),
                     Vector{T}(undef, n),
                     Vector{T}(undef, n),
                     Vector{T}(undef, n),
                     T(0),
                     Vector{T}(undef, m + 1),
                     :RUNNING,
                     Vector{T}(undef, m + 1)
                     )
        if opt.m > 0
            opt.dual = CCSAState(opt.m, # number of Lagrange multipliers
                                 0, # no inequality constraints
                                 λ -> (T[], T[]),
                                 ones(T, opt.m), # initial feasible point for Lagrange multipliers
                                 lb = zeros(T, opt.m))
        end
        return opt
    end
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

function step!(opt::CCSAState)
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