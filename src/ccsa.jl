mutable struct CCSAState{T<:AbstractFloat}
    n::Int # number of variables
    m::Int # number of constraints
    lb::AbstractVector{T} # n
    ub::AbstractVector{T} # n
    f_and_∇f::Function # f(x) = (m+1, (m+1) x n linear operator)
    ρ::AbstractVector{T} # m + 1
    σ::AbstractVector{T} # n
    x::AbstractVector{T} # current best guess
    xtol_rel::T
    max_iters::Integer
    # Above are essential
    # Below are temp
    fx::AbstractVector{T} # (m+1) x 1 output at x
    ∇fx # (m+1) x n linear operator of gradient at x
    a::AbstractVector{T} # n
    b::AbstractVector{T} # n
    Δx_zeroed::AbstractVector{T} # n
    Δx::AbstractVector{T} # n
    gλ::T
    ∇gλ::AbstractVector{T} # m+1, the extra first dimension is for convince
    Δx_last::AbstractVector{T}
    iters::Integer
    function CCSAState(
        n::Integer,
        m::Integer,
        f_and_∇f::Function,
        x::AbstractVector{T}=zeros(T,n);
        ρ::AbstractVector{T}=ones(T,m+1),
        σ::AbstractVector{T}=ones(T,n),
        lb::AbstractVector{T}=fill(typemin(T), n),
        ub::AbstractVector{T}=fill(typemax(T), n),
        xtol_rel::T=1e-4,
        max_iters::Integer=typemax(Int)
        ) where {T<:AbstractFloat}
        fx, ∇fx = f_and_∇f(x)
        return new{T}(n,m,lb,ub,f_and_∇f,ρ,σ,x,xtol_rel,max_iters,fx,∇fx, Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n),T(0), Vector{T}(undef, m+1), Vector{T}(undef, n),0)
    end
end
# Returns the dual function g(λ) and ∇g(λ)
function dual_func!(λ::AbstractVector{T}, st::CCSAState{T}) where {T}
    λ_all = CatView([one(T)], λ)
    st.a .= dot(st.ρ, λ_all) ./ (2 .* (st.σ).^2)
    mul!(st.b, st.∇fx', λ_all)
    @. st.Δx = clamp(-st.b / (2 * st.a), -st.σ, st.σ)
    @. st.Δx = clamp(st.Δx, st.lb-st.x, st.ub-st.x)
    st.gλ = dot(λ_all, st.fx) + sum((st.a) .* (st.Δx).^2 .+ (st.b) .* (st.Δx))
    mul!(st.∇gλ, st.∇fx, st.Δx)
    st.∇gλ .+= st.fx
    st.∇gλ .+= st.ρ.*sum((st.Δx).^2 ./ (2 .* (st.σ).^2))
    return [st.gλ], (@view st.∇gλ[2:st.m+1,:])'
end
function optimize_simple(opt::CCSAState)
    monotonic=BitVector(undef,opt.n)
    while opt.iters<opt.max_iters
        opt.fx, opt.∇fx = opt.f_and_∇f(opt.x)
        opt.a .= opt.ρ[1]/2 ./ (opt.σ).^2
        opt.b .= opt.∇fx'
        while true
            @. opt.Δx = clamp(-opt.b / (2 * opt.a), -opt.σ, opt.σ)
            @. opt.Δx = clamp(opt.Δx, opt.lb-opt.x, opt.ub-opt.x)
            opt.gλ = opt.fx[1] + sum(@. (opt.a)*(opt.Δx)^2+(opt.b)*(opt.Δx))
            if opt.gλ >= opt.f_and_∇f(opt.x+opt.Δx)[1][1]
                break
            end
            opt.ρ *= 2
            opt.a *= 2
        end
        opt.ρ /=2
        monotonic .= signbit.(opt.Δx_last) .== signbit.(opt.Δx) 
        opt.σ[monotonic] *= 2 
        opt.σ[.!monotonic] /= 2
        opt.Δx_last .= opt.Δx
        opt.x .= opt.x .+ opt.Δx
        if norm(opt.Δx,Inf) < opt.xtol_rel
            break
        end
    end
end
function inner_iterations(opt::CCSAState)
    max_problem(λ)= begin 
        result=dual_func!(λ,opt) 
        -result[1], -result[2]
    end
    g₍ₓ₎=similar(opt.fx)
    conservative=BitVector(undef,opt.m+1)
    opt_again=CCSAState(opt.m,0,max_problem,zeros(opt.m),lb=zeros(opt.m)) 
    while true
        opt.fx, opt.∇fx = opt.f_and_∇f(opt.x)
        optimize_simple(opt_again) 
        dual_func!(opt_again.x,opt)
        mul!(g₍ₓ₎,opt.∇fx,opt.Δx)
        g₍ₓ₎ .+= opt.fx
        g₍ₓ₎ .+= 2 .* (opt.ρ) .* sum(abs2,(opt.Δx)./(opt.σ))
        conservative .= ( g₍ₓ₎ .>= opt.f_and_∇f(opt.x+opt.Δx)[1])
        if all(conservative)
            break
        end
        opt.ρ[.!conservative] *= 2
    end
end

function optimize(opt::CCSAState;callback=nothing)
    if opt.m==0
        optimize_simple(opt)
    end
    monotonic=BitVector(undef,opt.n)
    while opt.iters<opt.max_iters
        inner_iterations(opt)
        monotonic .= signbit.(opt.Δx_last) .== signbit.(opt.Δx) 
        opt.σ[monotonic] *= 2 
        opt.σ[.!monotonic] /= 2
        opt.ρ /=2
        opt.Δx_last .= opt.Δx
        opt.x .= opt.x .+ opt.Δx
        if norm(opt.Δx,Inf) < opt.xtol_rel
            break
        end
        opt.iters+=1
        if callback!=nothing
            callback()
        end
    end
end
