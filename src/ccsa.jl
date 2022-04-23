# Full description of problem
# TODO: proper generic typing
# TODO: don't stuff everything into a single struct (do this later, not now)
mutable struct CCSAState{T<:Real}
    # describes problem
    n::Int # number of variables
    m::Int # number of constraints
    lb::AbstractVector{T} # n
    ub::AbstractVector{T} # n
    f_and_grad::Function # f(x) = (output, (m+1) x n linear operator)

    # state needed for inner iteration
    ρ::AbstractVector{T} # m + 1
    σ::AbstractVector{T} # n
    x::AbstractVector{T} # current best guess
    fx::T # (m+1) x 1 output at x
    gradx # (m+1) x n linear operator of gradient at x

    # arrays used by dual_func
    a::AbstractVector{T} # n
    b::AbstractVector{T} # n
    dx_unclamped::AbstractVector{T} # n
    dx_zeroed::AbstractVector{T} # n
    dx::AbstractVector{T} # n
    gradλ::AbstractVector{T} # m
end

# returns (g(λ), gradient of g(λ))
# don't worry about allocations for now
function dual_func!(λ::AbstractVector{T}, st::CCSAState) where {T}
    λ_all = CatView(1, λ)

    # TODO: handle σ = 0 if necessary
    @. st.a = 1 / (2 * st.σ^2) * dot(st.ρ, λ_all)
    mul!(st.b, st.gradx', λ_all)

    S = zero(T)

    for j in 1:m
        st.dx_unclamped[j] = -b[j] / 2 * a[j]
        st.dx[j] = clamp(dx[j], -σ[j], σ[j])
        st.dx_zeroed[j] = abs(dx[j]) < σ ? dx[j] : zero(T)
        S += b[j]^2 / (8 * a[j]^2 * σ[j]^2)
    end

    gλ = dot(λ, fx) + sum(a[j] * dx[j] + b[j] * dx[j]^2 for j in 1:n)
    mul!(gradλ, st.gradx, st.dx_zeroed)
    @. gradλ += ρ * S

    gλ, gradλ
end
