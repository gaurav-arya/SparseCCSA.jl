# Provides all necessary parameters to describe the approximate problem
# (to be solved by inner iterations on the dual of this problem)
struct DualData{T}
    count::Int
    n::Int

    # length n arrays (over dimensions). Do I want to use StaticArays?
    x::AbstractVector{T} 
    lb::AbstractVector{T}
    ub::AbstractVector{T} # do I need these? Can I keep just σ? I guess these clip at X.
    σ::AbstractVector{T}

    # (m+1) x n matrix
    dfdx_all # matrix-free Jacobian of objective + contraints

    # length (m+1) arrays (over objective + constraints)
    fval_all::T
    ρ_all::T
end

struct DualWork{T}
    a::AbstractVector{T} # length n
    b::AbstractVector{T} # length n
end

# evaluates g(y) and populates grad with ∇g(y)
function dual_func!(y::AbstractVector{T}, grad::AbstractVector{T}, d::DualData{T}, dw::DualWork{T}) where {T}
    y_all = vcat(1, y) # TODO: use views

    # assume σ > 0 for now
    u = dot(d.ρ_all, d.y_all)
    @. dw.a = 1 / (2 * σ^2) * u
    mul!(dw.b, dfdx_all', y_all)

    s1 = sum(dw.b[j]^2 / dw.a[j] for j in 1:n)
    val = dot(y_all, fval_all) - s1/4

    s2 = sum(dw.b[j]^2 / (dw.a[j]^2 * σ[j]^2) for j in 1:n)
    @. dw.b /= 2 * dw.a
    mul!(grad, dfdx_all, dw.b)
    @. grad -= ρ / (8 * s2)

    val
end
