mutable struct CCSAState{T<:Real}
    n::Int # number of variables
    m::Int # number of constraints
    lb::AbstractVector{T} # n
    ub::AbstractVector{T} # n
    f_and_∇f::Function # f(x) = (m+1, (m+1) x n linear operator)
    ρ::AbstractVector{T} # m + 1
    σ::AbstractVector{T} # n
    x::AbstractVector{T} # current best guess
    xtol_rel::Float64
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
    x⁻¹::AbstractVector{T}
    function CCSAState(n,m,f_and_fgrad,ρ,σ,x)
        return new{Float64}(n,m,ones(n)*(-2^20),ones(n)*(2^20),f_and_fgrad,ρ,σ,x,1e-5,zeros(m+1),zeros(m+1,n),zeros(n),zeros(n),zeros(n),ones(n),0,zeros(m+1),zeros(n))
    end
    function CCSAState(n,m,f_and_fgrad,ρ,σ,x,lb)
        return new{Float64}(n,m,lb,zeros(n)*(2^20),f_and_fgrad,ρ,σ,x,1e-5,zeros(m+1),zeros(m+1,n),zeros(n),zeros(n),zeros(n),ones(n),0,zeros(m+1),zeros(n))
    end
    # TODO: Does Julia has something like "Inf" ?
end
#= Gaurav's dual_func!
function dual_func!(λ::AbstractVector{T}, st::CCSAState) where {T}
    λ_all = CatView(1, λ)

    # TODO: handle σ = 0 if necessary
    @. st.a = 1 / (2 * st.σ^2) * dot(st.ρ, λ_all)
    mul!(st.b, st.gradx', λ_all)

    S = zero(T)
    for j in 1:st.n
        st.dx_unclamped[j] = -st.b[j] / 2 * st.a[j]
        st.dx[j] = clamp(st.dx_unclamped[j], -st.σ[j], st.σ[j])
        st.dx_zeroed[j] = abs(st.dx_unclamped[j]) < σ ? st.dx_unclamped[j] : zero(T)
        S += b[j]^2 / (8 * a[j]^2 * σ[j]^2)
    end

    gλ = dot(λ_all, st.fx) + sum(st.a[j] * st.dx[j] + st.b[j] * st.dx[j]^2 for j in 1:st.n)
    mul!(st.gradλ, st.gradx, st.dx_zeroed)
    @. st.gradλ += ρ * S

    gλ, @view st.gradλ[2:end]
end
=#
function dual_func!(λ::AbstractVector{T}, st::CCSAState) where {T}
    st.fx,st.∇fx=st.f_and_∇f(st.x)
    λ_all = CatView([1.0],λ)
    # TODO: handle σ = 0 if necessary
    st.a .= 1 ./ (2 .* (st.σ).^2) * dot(st.ρ, λ_all)
    mul!(st.b, st.∇fx', λ_all)
    @. st.Δx = clamp(-st.b / 2 * st.a, -st.σ, st.σ)
    @. st.Δx = clamp(st.Δx, st.lb-st.x, st.ub-st.x)
    st.gλ = dot(λ_all, st.fx) + sum((st.a) .* (st.Δx).^2 .+ (st.b) .* (st.Δx))
    mul!(st.∇gλ, st.∇fx, st.Δx)
    st.∇gλ .+= st.ρ.*sum((st.Δx).^2 ./ (2 .* (st.σ).^2))
    return [[st.gλ],st.∇gλ[2:st.m+1,:]'] #[1维vector，m维vector换成矩阵]
    # What are used in this?
    # st.f_and_∇f, st.x, λ, st.σ, st.ρ, 
end
function optimize_simple(opt::CCSAState)
    while true
        while true
            #println("   inner: Current ρ: $(opt.ρ)")
            println("           simple Current ρ/σ: $(opt.ρ/opt.σ)")
            println("           simple Current g(λ): $(opt.gλ)")
            println("           simple Current f(x+Δx): $(opt.f_and_∇f(opt.x+opt.Δx)[1][1][1])")
            dual_func!(Float64[], opt)
            if opt.gλ >= opt.f_and_∇f(opt.x+opt.Δx)[1][1]
                break
            end
            opt.ρ[1] *= 2
        end
        
        opt.ρ[1] *= 0.5
        update =  sign.(opt.x - opt.x⁻¹).*sign.(opt.Δx)
        for j in 1:opt.n
            if update[j] == 1
                opt.σ[j] *= 2.0
            elseif update[j] == -1
                opt.σ[j] *= 0.5
            end
        end 
        opt.x⁻¹ .= opt.x
        opt.x .= opt.x .+ opt.Δx

        #println("\nSuppose here is a callback")
        println("       Simple Current x: $(opt.x)")
        #println("Current σ: $(opt.σ)")

        if norm(opt.Δx) < opt.xtol_rel
            println("       Simple outer loop break now")
            break
        end
    end
    return nothing
end
function inner_iterations(opt::CCSAState)
    ρ_again=[1.0]
    σ_again=copy(opt.σ)
    while true
        opt_again=CCSAState(opt.n,0,λ->dual_func!(λ,opt),ρ_again,σ_again,zeros(opt.n),zeros(opt.n)) 
        optimize_simple(opt_again) 
        # 现在优化完了，找到了最好的λ，怎么回去找Δx？？？
        # 在跑一次dual_func
        # 因为新建state是copy出去了这个function，不会改变原来的值？？
        dual_func!(opt_again.x,opt)
        #λ=opt_again.x
        #没有constraint的时候，g₍ₓ₎就是g₍λ₎
        #现在不是了，现在g₍ₓ₎是m+1维，g₍λ₎是一维
        #计算g(x)
        g₍ₓ₎=copy(opt.fx)
        mul!(g₍ₓ₎,opt.∇fx,opt.Δx)
        g₍ₓ₎ .+= 0.5 .* (opt.ρ).^2 .* sum(abs2,(opt.Δx)./(opt.σ))
        println("   inner Current ρ/σ: $(opt.ρ/opt.σ)")
        println("   inner Current f(x+Δx): $(opt.f_and_∇f(opt.x+opt.Δx)[1])")
        println("   inner Current g(x): $(g₍ₓ₎)")
        conservative = ( g₍ₓ₎ .>= opt.f_and_∇f(opt.x+opt.Δx)[1])
        if all(conservative)
            break
        end
        opt.ρ[.!conservative] *= 2
    end
end
function optimize(opt::CCSAState)
    if opt.m==0
        optimize_simple(opt)
        return nothing
    end
    test=dual_func!(zeros(opt.m),opt)
    while true
        inner_iterations(opt)

        opt.ρ .*= 0.5
        update =  sign.(opt.x - opt.x⁻¹).*sign.(opt.Δx)
        for j in 1:opt.n
            if update[j] == 1
                opt.σ[j] *= 2.0
            elseif update[j] == -1
                opt.σ[j] *= 0.5
            end
        end 
        opt.x⁻¹ .= opt.x
        opt.x .= opt.x .+ opt.Δx

        println("Current x: $(opt.x)")
        #println("Current σ: $(opt.σ)")
        if norm(opt.Δx) < opt.xtol_rel
            break
        end
    end
end
