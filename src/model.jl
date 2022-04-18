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
