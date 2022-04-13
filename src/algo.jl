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

function inner_iteration(opt::CCSA_Opt, xᵏ::AbstractVector)
    ∇f_xᵏ = Array{Float64,2}(undef, opt.m + 1, opt.n)
    f_xᵏ = map(i -> opt.f[i](xₖ, @view ∇f_xᵏ[i, :]), 1:opt.m+1)
    function approx_func(x::AbstractVector)
        diff = x - xₖ
        diff²_σ²_sum_2 = sum((diff ./ opt.σ) .^ 2) / 2
        g = f_xᵏ + ∇f_xᵏ * diff + opt.ρ * diff²_σ²_sum_2
        g
    end

    while true
        xᵏ⁺¹ = similar(xₖ) # TODO: optimize inner_iteration dual
        g_xᵏ⁺¹ = approx_func(xᵏ⁺¹)
        f_xᵏ⁺¹ = map(fᵢ -> fᵢ(xᵏ⁺¹, []), opt.f)
        conservative = g_xᵏ⁺¹ .>= f_xᵏ⁺¹
        if all(conservative)
            break
        end
        opt.ρ[.!conservative] *= 2
    end

    return xᵏ⁺¹
end

function optimize(opt::CCSA_Opt, x⁰::AbstractVector)
    xᵏ⁻¹ = copy(x⁰) # TODO
    xᵏ = copy(x⁰)
    while true
        xᵏ⁺¹ = inner_iteration(opt, xᵏ)
        opt.ρ *= 0.5
        signᵏ = sign.(xᵏ - xᵏ⁻¹)
        signᵏ⁺¹ = sign.(xᵏ⁺¹ - xᵏ)
        update = signᵏ .* signᵏ⁺¹
        map(1:opt.n) do j
            if update[j] == 1
                opt.σ[j] *= 2.0
            elseif update[j] == -1
                opt.σ[j] *= 0.5
            end
        end
        xᵏ⁻¹ = xᵏ
        xᵏ = xᵏ⁺¹
        if norm(xᵏ - xᵏ⁻¹) < opt.xtol_rel
            break
        end
    end
    return xᵏ
end
