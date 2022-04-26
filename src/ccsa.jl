mutable struct CCSAState{T<:Real}
    n::Int # number of variables
    m::Int # number of constraints
    lb::AbstractVector{T} # n
    ub::AbstractVector{T} # n
    f_and_∇f::Function # f(x) = (output, (m+1) x n linear operator)
    ρ::AbstractVector{T} # m + 1
    σ::AbstractVector{T} # n
    x::AbstractVector{T} # current best guess
    xtol_rel::Float64
    fx::AbstractVector{T} # (m+1) x 1 output at x
    ∇fx # (m+1) x n linear operator of gradient at x
    a::AbstractVector{T} # n
    b::AbstractVector{T} # n
    Δx_zeroed::AbstractVector{T} # n
    Δx::AbstractVector{T} # n
    gλ::T
    ∇gλ::AbstractVector{T} # m
    xᵏ⁻¹::AbstractVector{T}
    function CCSAState(n,m,f_and_fgrad,ρ,σ,x)
        return new{Float64}(n,m,ones(n)*(-2^20),zeros(n)*(2^20),f_and_fgrad,ρ,σ,x,1e-5,zeros(m+1),zeros(m+1,n),zeros(n),zeros(n),zeros(n),ones(n),0,zeros(m),zeros(n))
    end
    function CCSAState(n,m,f_and_fgrad,ρ,σ,x,lb)
        return new{Float64}(n,m,lb,zeros(n)*(2^20),f_and_fgrad,ρ,σ,x,1e-5,zeros(m+1),zeros(m+1,n),zeros(n),zeros(n),zeros(n),ones(n),0,zeros(m),zeros(n))
    end
end

function dual_func!(λ::AbstractVector{T}, st::CCSAState) where {T}
    st.fx,st.∇fx=st.f_and_∇f(st.x)
    λ_all = CatView([1.0],λ)
    # TODO: handle σ = 0 if necessary
    st.a .= 1 ./ (2 .* (st.σ).^2) * dot(st.ρ, λ_all)
    mul!(st.b, st.∇fx', λ_all)
    @. st.Δx = clamp(-st.b / 2 * st.a, -st.σ, st.σ)
    @. st.Δx = clamp(st.Δx, st.lb-st.x, st.ub-st.x)
    st.gλ = dot(λ_all, st.fx) + sum((st.a) .* (st.Δx).^2 .+ (st.b) .* (st.Δx))
    TMP=Vector{T}(undef,m+1)
    mul!(TMP, st.∇fx, st.Δx)
    st.∇gλ = TMP[2:m+1]
    st.∇gλ .+= st.ρ.*sum((st.Δx).^2 ./ (2 .* (st.σ).^2))
    return [st.gλ,st.∇gλ]
end
