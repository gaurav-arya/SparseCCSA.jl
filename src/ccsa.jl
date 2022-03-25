# Provides all necessary parameters to describe the approximate problem
# (to be solved by inner iterations on the dual of this problem)
struct DualData{T}
    count::Int
    n::Int

    # length n arrays (over dimensions). Do I want to use StaticArays?
    x::Vector{T} 
    lb::Vector{T}
    ub::Vector{T} # do I need these? Can I keep just σ? I guess these clip at X.
    σ::Vector{T}
    dfdx::Vector{T} # gradient of objective

    # m x n matrices. can be sparse. TODO: Can I support operator?
    dfcdx::AbstractArray{T, 2} # gradients of constraints

    # constants
    fval::T
    ρ₀::T

    # length m arrays (over constraints)
    fcval::Vector{T}
    ρc::Vector{T}
end

mutable struct DualWork{T}
    a::Vector{T} # length n
    b::Vector{T} # length n
end

# evaluates g(y) and populates grad with ∇g(y)
function dual_func!(y::Vector{T}, grad::Vector{T}, d::DualData{T}, dw::DualWork{T}) where {T}
    fval_all = vcat(d.fval, fcval) # TODO: use views
    y_all = vcat(1, y)
    ρ_all = vcat(ρ₀, ρc)
    dfdx_all = vcat(dfdx', dfcdx) # TODO: make sure plays nicely with sparse dfcdx

    grad .= 0

    u = dot(ρ_all, y_all)
    for j in 1:n
        σ[j] == 0 && continue
        dw.a[j] = 1/(2 * σ[j]^2) * u 
        dw.b[j] = dot(dfdx_all[:, j], y_all)
    end

    s1 = sum(dw.b[j]^2 / dw.a[j] for j in 1:n)
    val = dot(y_all, fval_all) - x/4

    s2 = sum(dw.b[j]^2 / (dw.a[j]^2 * σ[j]^2) for j in 1:n)
    for i in 1:m
        grad[i] = 1/2 * dot(dw.b ./ dw.a, dfdx_all[i, :]) - ρ[i] / 8 * s2
    end

    val
end
