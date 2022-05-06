mutable struct CCSAState{T<:AbstractFloat}
    n::Integer # number of variables > 0
    m::Integer # number of inequality constraints ≥ 0
    lb::AbstractVector{T} # n lower bounds
    ub::AbstractVector{T} # n upper bounds
    f_and_∇f::Function # f(x) = (m+1, (m+1) x n linear operator)
    ρ::AbstractVector{T} # m + 1 penality weight
    σ::AbstractVector{T} # n radius of trust region
    x::AbstractVector{T} # current best feasible point
    xtol_rel::T # relative tolerence

    fx::AbstractVector{T} # (m+1) x 1 function values at x
    ∇fx::AbstractVecOrMat{T} # (m+1) x n linear operator of Jacobian at x
    a::AbstractVector{T} # n
    b::AbstractVector{T} # n
    Δx::AbstractVector{T} # n xᵏ⁺¹ - xᵏ
    Δx_last::AbstractVector{T} # n xᵏ - xᵏ⁻¹
    gλ::T # Lagrange dual function value
    ∇gλ::AbstractVector{T} # m Lagrange dual function gradient

    function CCSAState(
        n::Integer, # number of variables
        m::Integer, # number of inequality constraints
        f_and_∇f::Function,
        penality_weight::AbstractVector{T}, # (m + 1) penality weight ρ
        trust_radius::AbstractVector{T}, # n radius of trust region σ
        x₀::AbstractVector{T}; # initial feasible point
        lb::AbstractVector{T}=fill(typemin(T), n), # lower bounds, default -Inf
        ub::AbstractVector{T}=fill(typemax(T), n), # upper bounds, default Inf
        xtol_rel::T=T(1e-5) # relative tolerence
    ) where {T<:AbstractFloat}
        fx, ∇fx = f_and_∇f(x₀)
        new{T}(
            n,
            m,
            lb,
            ub,
            f_and_∇f,
            penality_weight,
            trust_radius,
            x₀,
            xtol_rel,
            fx,
            ∇fx,
            Vector{T}(undef, n),
            Vector{T}(undef, n),
            Vector{T}(undef, n),
            Vector{T}(undef, n),
            T(0),
            Vector{T}(undef, m)
        )
    end
end

# Returns the dual function g(λ) and ∇g(λ)
function dual_func!(λ::AbstractVector{T}, st::CCSAState{T}) where {T}
    λ_all = CatView([one(T)], λ)
    st.a .= dot(st.ρ, λ_all) ./ (2 .* st.σ .^ 2)
    mul!(st.b, st.∇fx', λ_all)
    @. st.Δx = clamp(-st.b / (2 * st.a), -st.σ, st.σ)
    @. st.Δx = clamp(st.Δx, st.lb - st.x, st.ub - st.x)
    st.gλ = dot(λ_all, st.fx) + sum(@. st.a * st.Δx^2 + st.b * st.Δx)
    mul!(st.∇gλ, st.∇fx[2:end, :], st.Δx)
    st.∇gλ += st.fx[2:end] + sum(abs2, st.Δx ./ st.σ) / 2 * st.ρ[2:end]
    return [st.gλ], st.∇gλ'
end

function update_trust_region_and_penalty!(opt::CCSAState)
    opt.ρ *= 0.5 # reduce penality wight to allow larger subsequent steps if possible
    monotonic = signbit.(opt.Δx_last) .== signbit.(opt.Δx) # signbit avoid multiplication
    opt.σ[monotonic] *= 2 # double trust region if xⱼ moves monotomically
    opt.σ[.!monotonic] *= 0.5 # shrink trust region if xⱼ oscillates
    return
end

# optimize problem with no constraint
function optimize_simple(opt::CCSAState{T}) where {T}
    while true
        while true
            dual_func!(T[], opt)
            if opt.gλ ≥ opt.f_and_∇f(opt.x + opt.Δx)[1][1] # check conservative
                break
            end
            opt.ρ *= 2
        end
        update_trust_region_and_penalty!(opt)
        opt.x += opt.Δx
        if norm(opt.Δx) < opt.xtol_rel
            return
        end
        opt.Δx_last = opt.Δx
        opt.fx, opt.∇fx = opt.f_and_∇f(opt.x)
    end
end

function inner_iterations(opt::CCSAState{T}) where {T}
    # input: opt::CCSAState
    # output: update opt.Δx
    # opt.Δx is the solution of the dual problem
    # i.e. max{min{g0(x)+λ1g1(x)...}}=max{g(λ)}
    # gi are constructed by opt.ρ/opt.σ
    ρ_again = [one(T)]
    σ_again = ones(T, opt.m)
    while true
        opt.fx, opt.∇fx = opt.f_and_∇f(opt.x)
        Lagrange_dual(λ) = begin
            gλ, ∇gλ = dual_func!(λ, opt)
            -gλ, -∇gλ # minus signs used to change max problem to min problem
        end
        opt_again = CCSAState( # Lagrange dual problem
            opt.m, # number of Lagrange multipliers
            0, # no inequality constraints
            Lagrange_dual, # negative Lagrange dual function and gradient
            ρ_again, # penality weight
            σ_again, # radii of trust regions
            ones(T, opt.m), # initial feasible point
            lb=zeros(T, opt.m) # nonnegative Lagrange multipliers for inequality constraints
        )
        optimize_simple(opt_again)
        λ = opt_again.x # result of Lagrange dual problem
        dual_func!(λ, opt)
        gᵢxᵏ⁺¹ = similar(opt.fx) # TODO: avoid heap allocation
        mul!(gᵢxᵏ⁺¹, opt.∇fx, opt.Δx)
        gᵢxᵏ⁺¹ += opt.fx + sum(abs2, opt.Δx ./ opt.σ) * 0.5 * opt.ρ
        fᵢxᵏ⁺¹ = opt.f_and_∇f(opt.x + opt.Δx)[1]
        conservative = gᵢxᵏ⁺¹ .≥ fᵢxᵏ⁺¹
        if all(conservative)
            return
        end
        opt.ρ[.!conservative] *= 2 # increase ρ until achieving conservative approxmation
    end
end

function optimize(opt::CCSAState{T}) where {T}
    if opt.m == 0
        optimize_simple(opt)
        return
    end
    while true
        inner_iterations(opt)
        update_trust_region_and_penalty!(opt)
        opt.x += opt.Δx
        if norm(opt.Δx) < opt.xtol_rel # TODO: adjust stop criteria
            return
        end
        opt.Δx_last = opt.Δx
    end
end
