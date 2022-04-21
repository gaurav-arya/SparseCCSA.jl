using NLPModels, LinearAlgebra

mutable struct CCSAModel{T,S} <: AbstractNLPModel{T,S}
    meta::NLPModelMeta{T,S}
    counters::Counters
    xtol_rel::T # x relative tolerence
    ρ₀::T # penality weight for obj f₀
    ρᵢ::AbstractVector{T} # [ncon] penality weight for contraints fᵢ
    σ::AbstractVector{T} # [nvar] radius of trust region
end

function CCSAModel(
    nvar::Int, # number of variables
    ncon::Int, # number of constraints
    x0::AbstractVector{T}, # initial point (must be feasible)
    lvar::AbstractVector{T}, # vector of lower bounds
    uvar::AbstractVector{T}, # vector of upper bounds
    xtol_rel::T, # x relative tolerence
    ρ₀::T, # penality weight for obj f₀
    ρᵢ::AbstractVector{T}, # [ncon] penality weight for contraints fᵢ
    σ::AbstractVector{T} # [nvar] radius of trust region
) where {T}
    CCSAModel(
        NLPModelMeta(nvar;
            x0=x0,
            lvar=lvar,
            uvar=uvar,
            ncon=ncon),
        Counters(),
        xtol_rel,
        ρ₀,
        ρᵢ,
        σ)
end

function dualModel(
    nvar::Int, # number of variables
    x0::AbstractVector{T}, # initial point (must be feasible)
    xtol_rel::T, # x relative tolerence
    ρ₀::T, # penality weight for obj f₀
    σ::AbstractVector{T} # [nvar] radius of trust region
) where {T}
    CCSAModel(
        NLPModelMeta(nvar;
            x0=x0,
            lvar=zeros(T, nvar)),
        Counters(),
        xtol_rel,
        ρ₀,
        T[],
        σ)
end

function inner_iter(model::CCSAModel, xᵏ::AbstractVector{T}) where {T}
    @lencheck model.meta.nvar xᵏ
    σ² = model.σ .^ 2
    x_min = max.(xᵏ - model.σ, model.meta.lvar) # trust region lower bound
    x_max = min.(xᵏ + model.σ, model.meta.uvar) # trust region upper bound
    if model.meta.ncon == 0 # a problem without contraints (probably dual)
        f₀xᵏ, ∇f₀xᵏ = NLPModels.objcons(model, xᵏ)
        x̂ = xᵏ - ∇f₀xᵏ .* σ² / model.ρ₀
        return @. min(max(x̂, x_min), x_max)
    end
    f₀xᵏ = NLPModels.obj(model, xᵏ)
    ∇f₀xᵏ = NLPModels.grad(model, xᵏ)
    fᵢxᵏ = NLPModels.cons(model, xᵏ)
    Jfᵢxᵏ = NLPModels.jac(model, xᵏ)
    while true
        @eval function approx_objcons(x) # gᵢ around xᵏ, i ∈ {0,1,...,m}
            # approximation of obj f₀ and constraints fᵢ around xᵏ
            # 1st order Taylor + weighted quadratic penality for step size
            Δx = x - xᵏ
            penality = sum(@. (Δx / model.σ)^2) / 2
            g₀x = f₀xᵏ + dot(∇f₀xᵏ, Δx) + model.ρ₀ * penality
            gᵢx = fᵢxᵏ + Jfᵢxᵏ * Δx + model.ρᵢ * penality
            g₀x, gᵢx
        end
        @eval function x_dual(λ::AbstractVector)
            x̂ = xᵏ - (∇f₀xᵏ + Jfᵢxᵏ' * λ) ./ (model.ρ₀ + dot(λ, model.ρᵢ)) .* σ²
            @. min(max(x̂, x_min), x_max) # bounded by trust region
        end
        dual = dualModel(
            model.meta.ncon, # number of λᵢ == number of primal constraints
            ones(T, model.meta.ncon), # initial feasible point
            model.xtol_rel, # relative tolerence
            1.0, # ρ₀
            ones(T, model.meta.ncon)) # σ
        @eval function NLPModels.objcons(
            dualModel::CCSAModel,
            λ::AbstractVector
        )
            # obj and gradient for dual problem
            @lencheck dualModel.meta.nvar λ
            increment!(model, :neval_obj)
            increment!(model, :neval_grad)
            x = x_dual(λ) # x(λ)
            g₀x, gᵢx = approx_objcons(x)
            g₀x + dot(λ, gᵢx), gᵢx
        end
        λ_optimal = optimize(dual)
        xᵏ⁺¹ = x_dual(λ_optimal)
        f₀xᵏ⁺¹ = NLPModels.obj(model, xᵏ⁺¹)
        fᵢxᵏ⁺¹ = NLPModels.cons(model, xᵏ⁺¹)
        g₀xᵏ⁺¹, gᵢxᵏ⁺¹ = approx_objcons(xᵏ⁺¹)
        conservative₀ = g₀xᵏ⁺¹ ≥ f₀xᵏ⁺¹
        conservativeᵢ = gᵢxᵏ⁺¹ .≥ fᵢxᵏ⁺¹
        if conservative₀
            if all(conservativeᵢ)
                return xᵏ⁺¹
            end
        else
            model.ρ₀ *= 2
        end
        model.ρᵢ[.!conservativeᵢ] *= 2
    end
end

function optimize(model::CCSAModel)
    xᵏ = copy(model.meta.x0)
    xᵏ⁺¹ = inner_iter(model, xᵏ)
    Δxᵏ = xᵏ⁺¹ - xᵏ
    signᵏ = signbit.(Δxᵏ) # used to check if x moves monotomically or oscillates
    while any(@. abs(Δxᵏ / xᵏ) > model.xtol_rel)
        model.ρ₀ /= 2 # geometrically decrease penality weight to
        model.ρᵢ /= 2 # allow larger subsequent step if possible
        xᵏ = xᵏ⁺¹
        xᵏ⁺¹ = inner_iter(model, xᵏ)
        Δxᵏ = xᵏ⁺¹ - xᵏ
        signᵏ⁺¹ = signbit.(Δxᵏ)
        monotonic = signᵏ .== signᵏ⁺¹ # avoid multiplication
        model.σ[monotonic] *= 2 # double trust region if xⱼ moves monotomically
        model.σ[.!monotonic] /= 2 # shrink trust region if xⱼ oscillates
        signᵏ = signᵏ⁺¹
    end
    xᵏ⁺¹
end
