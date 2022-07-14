mutable struct CCSAState{T <: AbstractFloat}
    n::Integer # number of variables > 0
    m::Integer # number of inequality constraints ≥ 0
    lb::AbstractVector{T} # n lower bounds
    ub::AbstractVector{T} # n upper bounds
    f_and_∇f::Function # f(x) = (m+1, (m+1) x n linear operator)
    ρ::AbstractVector{T} # m + 1 penality weight
    σ::AbstractVector{T} # n radius of trust region
    x::AbstractVector{T} # current best feasible point
    xtol_rel::T # relative tolerence
    xtol_abs::T # absolute tolerence
    ftol_rel::T # relative tolerence
    ftol_abs::T # absolute tolerence
    max_iters::Integer # max number of iterations
    iters::Integer # interation count

    fx::AbstractVector{T} # (m+1) x 1 function values at x
    ∇fx::AbstractVecOrMat{T} # (m+1) x n linear operator of Jacobian at x
    a::AbstractVector{T} # n
    b::AbstractVector{T} # n
    Δx::AbstractVector{T} # n xᵏ⁺¹ - xᵏ
    Δx_last::AbstractVector{T} # n xᵏ - xᵏ⁻¹
    gλ::T # Lagrange dual function value
    ∇gλ::AbstractVector{T} # m Lagrange dual function gradient
    RET::Symbol
    fx_last::AbstractVector{T}
    dual::CCSAState{T} # Lagrange dual problem if the primal problem has constraints

    function CCSAState(n::Integer, # number of variables
                       m::Integer, # number of inequality constraints
                       f_and_∇f::Function,
                       x₀::AbstractVector{T}; # initial feasible point
                       ρ::AbstractVector{T} = ones(T, m + 1), # (m + 1) penality weight ρ
                       σ::AbstractVector{T} = ones(T, n), # n radius of trust region σ
                       lb::AbstractVector{T} = fill(typemin(T), n), # lower bounds, default -Inf
                       ub::AbstractVector{T} = fill(typemax(T), n), # upper bounds, default Inf
                       xtol_rel::T = T(1e-5), # relative tolerence
                       xtol_abs::T = T(1e-5), # relative tolerence
                       ftol_rel::T = T(1e-5), # relative tolerence
                       ftol_abs::T = T(1e-5), # relative tolerence
                       max_iters::Integer = typemax(Int64)
                       ) where {T <: AbstractFloat}
        fx, ∇fx = f_and_∇f(x₀)
        opt = new{T}(n,
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
# Returns the dual function g(λ) and ∇g(λ)
function dual_func!(λ::AbstractVector{T}, st::CCSAState{T}) where {T}
    λ_all = CatView([one(T)], λ)
    st.a .= dot(st.ρ, λ_all) ./ (2 .* st.σ .^ 2)
    mul!(st.b, st.∇fx', λ_all)
    for j in 1:(st.n)
        st.Δx[j] = clamp(-st.b[j] / (2 * st.a[j]), -st.σ[j], st.σ[j])
        st.Δx[j] = clamp(st.Δx[j], st.lb[j] - st.x[j], st.ub[j] - st.x[j])
    end
    st.gλ = dot(λ_all, st.fx) + sum(@. st.a * st.Δx^2 + st.b * st.Δx)
    st.∇gλ .= st.fx .+ sum(abs2, st.Δx ./ st.σ) / 2 .* st.ρ
    mul!(st.∇gλ, st.∇fx, st.Δx, true, true)
    return [st.gλ], (@view st.∇gλ[2:end, :])'
end

# optimize problem with no constraint
function optimize_simple(opt::CCSAState{T}) where {T}
    monotonic = BitVector(undef, opt.n)
    while opt.iters < 10#opt.max_iters
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

function inner_iterate(opt::CCSAState{T}) where {T}
    gᵢxᵏ⁺¹ = Vector{T}(undef, opt.m + 1)
    conservative = BitVector(undef, opt.m + 1)
    # arbitrary upper bound on inner iterations
    opt.fx, opt.∇fx = opt.f_and_∇f(opt.x) # expect this line to take up the most time. |Ax - b|^2 + λ |x|₁ <- x is length n, and there are n constraints.
    # normally doing A * x takes a lot longer than n time.
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

function optimize(opt::CCSAState{T}; callback = nothing) where {T}
    if opt.m == 0
        return optimize_simple(opt)
    end
    monotonic = BitVector(undef, opt.n)
    while true
        for i in 1:10
            inner_iterate(opt)
        end
        f = opt.fx[1]
        Δf = opt.f_and_∇f(opt.x+opt.Δx)[1][1] - f
        if opt.iters > opt.max_iters
            opt.RET = :MAX_ITERS
            return
        end
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