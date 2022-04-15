using LinearAlgebra

mutable struct CCSAOpt
    n::Int # number of variables
    m::Int # number of constraints
    lower_bounds::AbstractVector{Float64} # n
    upper_bounds::AbstractVector{Float64} # n
    xtol_rel::Float64
    f::Function # f(x) = output
    fgrad::Function # fgrad(x) = (m+1) x n linear operator
    ρ::AbstractVector{Float64} # m + 1
    σ::AbstractVector{Float64} # n
end

function inner_iterations(opt::CCSAOpt, xᵏ::AbstractVector)
    ∇f_xᵏ = Array{Float64,2}(undef, opt.m + 1, opt.n)
    f_xᵏ = map(i -> opt.f[i](xₖ, @view ∇f_xᵏ[i, :]), 1:opt.m+1) # TODO: adjust

    while true
        # Recursively call optimize with a new opt object
        # optimize g(y) using dual_func! 
        # once we find the best y, how do we find x^(k+1)?
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

function optimize(opt::CCSAOpt, x⁰::AbstractVector)
    xᵏ⁻¹ = copy(x⁰) # TODO
    xᵏ = copy(x⁰)
    while true
        xᵏ⁺¹ = inner_iterations(opt, xᵏ)
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
