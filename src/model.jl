using LinearAlgebra

mutable struct CCSAOpt{T}
    nvar::Int, # number of variables
    ncon::Int, # number of constraints
    x0::AbstractVector{T}, # initial point (must be feasible)
    lvar::AbstractVector{T}, # vector of lower bounds
    uvar::AbstractVector{T}, # vector of upper bounds
    xtol_rel::T, # x relative tolerence
    ρ::T, # [ncon + 1] penality weight for obj f₀
    σ::AbstractVector{T} # [nvar] radius of trust region
    fᵢJ::Function
end

function inner_iter(opt::CCSAOpt, xᵏ::AbstractVector{T}) where {T}
    σ² = opt.σ .^ 2
    x_min = max.(xᵏ - opt.σ, opt.lvar) # trust region lower bound
    x_max = min.(xᵏ + opt.σ, opt.uvar) # trust region upper bound
    if opt.ncon == 0 # a problem without contraints (probably dual)
        f₀xᵏ, ∇f₀xᵏ = NLPModels.objcons(opt, xᵏ)
        x̂ = xᵏ - ∇f₀xᵏ .* σ² / opt.ρ₀
        return @. min(max(x̂, x_min), x_max)
    end
    f₀xᵏ = NLPModels.obj(opt, xᵏ)
    ∇f₀xᵏ = NLPModels.grad(opt, xᵏ)
    fᵢxᵏ = NLPModels.cons(opt, xᵏ)
    Jfᵢxᵏ = NLPModels.jac(opt, xᵏ)
    while true
        @eval function approx_objcons(x) # gᵢ around xᵏ, i ∈ {0,1,...,m}
            # approximation of obj f₀ and constraints fᵢ around xᵏ
            # 1st order Taylor + weighted quadratic penality for step size
            Δx = x - xᵏ
            penality = sum(@. (Δx / opt.σ)^2) / 2
            g₀x = f₀xᵏ + dot(∇f₀xᵏ, Δx) + opt.ρ₀ * penality
            gᵢx = fᵢxᵏ + Jfᵢxᵏ * Δx + opt.ρᵢ * penality
            g₀x, gᵢx
        end
        @eval function x_dual(λ::AbstractVector)
            x̂ = xᵏ - (∇f₀xᵏ + Jfᵢxᵏ' * λ) ./ (opt.ρ₀ + dot(λ, opt.ρᵢ)) .* σ²
            @. min(max(x̂, x_min), x_max) # bounded by trust region
        end
        dual = dualModel(
            opt.meta.ncon, # number of λᵢ == number of primal constraints
            ones(T, opt.meta.ncon), # initial feasible point
            opt.xtol_rel, # relative tolerence
            1.0, # ρ₀
            ones(T, opt.meta.ncon)) # σ
        @eval function NLPModels.objcons(
            dualModel::CCSAModel,
            λ::AbstractVector
        )
            # obj and gradient for dual problem
            @lencheck dualModel.meta.nvar λ
            increment!(opt, :neval_obj)
            increment!(opt, :neval_grad)
            x = x_dual(λ) # x(λ)
            g₀x, gᵢx = approx_objcons(x)
            g₀x + dot(λ, gᵢx), gᵢx
        end
        λ_optimal = optimize(dual)
        xᵏ⁺¹ = x_dual(λ_optimal)
        f₀xᵏ⁺¹ = NLPModels.obj(opt, xᵏ⁺¹)
        fᵢxᵏ⁺¹ = NLPModels.cons(opt, xᵏ⁺¹)
        g₀xᵏ⁺¹, gᵢxᵏ⁺¹ = approx_objcons(xᵏ⁺¹)
        conservative₀ = g₀xᵏ⁺¹ ≥ f₀xᵏ⁺¹
        conservativeᵢ = gᵢxᵏ⁺¹ .≥ fᵢxᵏ⁺¹
        if conservative₀
            if all(conservativeᵢ)
                return xᵏ⁺¹
            end
        else
            opt.ρ₀ *= 2
        end
        opt.ρᵢ[.!conservativeᵢ] *= 2
    end
end

function optimize(opt::CCSAOpt)
    xᵏ = copy(opt.x0)
    xᵏ⁺¹ = inner_iter(opt, xᵏ)
    Δxᵏ = xᵏ⁺¹ - xᵏ
    signᵏ = signbit.(Δxᵏ) # used to check if x moves monotomically or oscillates
    while any(@. abs(Δxᵏ / xᵏ) > opt.xtol_rel)
        opt.ρ /= 2 
        # geometrically decrease penality weight to
        # allow larger subsequent step if possible
        xᵏ = xᵏ⁺¹
        xᵏ⁺¹ = inner_iter(opt, xᵏ)
        Δxᵏ = xᵏ⁺¹ - xᵏ
        signᵏ⁺¹ = signbit.(Δxᵏ)
        monotonic = signᵏ .== signᵏ⁺¹ # avoid multiplication
        opt.σ[monotonic] *= 2 # double trust region if xⱼ moves monotomically
        opt.σ[.!monotonic] /= 2 # shrink trust region if xⱼ oscillates
        signᵏ = signᵏ⁺¹
    end
    xᵏ⁺¹
end