using LinearAlgebra

mutable struct CCSA_Opt
    n::Int # number of variables
    m::Int # number of constraints
    lower_bounds::AbstractVector{Float64} # n
    upper_bounds::AbstractVector{Float64} # n
    xtol_rel::Float64
    f::AbstractVector{Function} # m + 1 (m constraints & 1 objective func)
    ρ::AbstractVector{Float64} # m + 1
    σ::AbstractVector{Float64} # n
    function CCSA_Opt(n_variable::Int, lower_bounds::AbstractVector{Float64},
        upper_bounds::AbstractVector{Float64}, xtol_rel::Float64,
        min_objective::Function, inequality_constraints::AbstractVector{Function},
        ρ::AbstractVector{Float64}, σ::AbstractVector{Float64})
        new(n_variable, size(inequality_constraints, 1),
            lower_bounds, upper_bounds, xtol_rel,
            vcat(inequality_constraints, min_objective),
            ρ, σ)
    end
end
