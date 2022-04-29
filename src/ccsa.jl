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
        return new{Float64}(n,m,lb,ones(n)*(2^20),f_and_fgrad,ρ,σ,x,1e-5,zeros(m+1),zeros(m+1,n),zeros(n),zeros(n),zeros(n),ones(n),0,zeros(m+1),zeros(n))
    end
    function CCSAState(n,m,f_and_fgrad,ρ,σ,x,lb,ub)
        return new{Float64}(n,m,lb,ub,f_and_fgrad,ρ,σ,x,1e-5,zeros(m+1),zeros(m+1,n),zeros(n),zeros(n),zeros(n),ones(n),0,zeros(m+1),zeros(n))
    end
    # TODO: Does Julia has something like "Inf" ?
end

# Returns the dual function g(λ) and ∇g(λ)
function dual_func!(λ::AbstractVector{T}, st::CCSAState) where {T}
    λ_all = CatView([one(T)], λ)
    st.a .= dot(st.ρ, λ_all) ./ (2 .* (st.σ).^2)
    mul!(st.b, st.∇fx', λ_all)
    @. st.Δx = clamp(-st.b / (2 * st.a), -st.σ, st.σ)
    @. st.Δx = clamp(st.Δx, st.lb-st.x, st.ub-st.x)
    st.gλ = dot(λ_all, st.fx) + sum((st.a) .* (st.Δx).^2 .+ (st.b) .* (st.Δx))
    mul!(st.∇gλ, st.∇fx, st.Δx)
    st.∇gλ .+= st.fx
    st.∇gλ .+= st.ρ.*sum((st.Δx).^2 ./ (2 .* (st.σ).^2))
    return [st.gλ], (@view st.∇gλ[2:st.m+1,:])' #[1维vector，m维vector换成矩阵]
    # What are used in this?
    # st.f_and_∇f, st.x, λ, st.σ, st.ρ, 
end
function optimize_simple(opt::CCSAState)
    while true
        opt.fx, opt.∇fx = opt.f_and_∇f(opt.x)
        while true
            dual_func!(Float64[], opt)
            #println("           simple Current g(x): $(opt.gλ)")
            #println("           simple Current f(x+Δx): $(opt.f_and_∇f(opt.x+opt.Δx)[1][1][1])")
            if opt.gλ >= opt.f_and_∇f(opt.x+opt.Δx)[1][1]
                break
            end
            opt.ρ[1] *= 2
        end
        #println("       Simple Current x: $(opt.x)")
        #println("       Simple Current Δx: $(opt.Δx)")
        #println("       Simple Current ρ: $(opt.ρ)")
        #println("       Simple Current σ: $(opt.σ)")
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

        if norm(opt.Δx) < opt.xtol_rel
            println("       Simple outer loop break now")
            break
        end
    end
    return nothing
end
function inner_iterations(opt::CCSAState)
# input: opt::CCSAState
# output: update opt.Δx
# opt.Δx is the solution of the dual problem
# i.e. max{min{g0(x)+λ1g1(x)...}}=max{g(λ)}
# gi are constructed by opt.ρ/opt.σ
    ρ_again=[1.0]
    σ_again=ones(opt.m)
    while true
        opt.fx, opt.∇fx = opt.f_and_∇f(opt.x)
        max_problem(λ)= begin 
            result=dual_func!(λ,opt) 
            -result[1], -result[2]
        end
        opt_again=CCSAState(opt.m,0,max_problem,ρ_again,σ_again,zeros(opt.m),zeros(opt.m)) 
        optimize_simple(opt_again) 
        dual_func!(opt_again.x,opt)
        println("   inner The optimal of dual λ: $(opt_again.x)")
        println("   inner The optimal of dual x: $(opt.x)")
        println("   inner The optimal of dual x+Δx: $(opt.x+opt.Δx)")
        g₍ₓ₎=similar(opt.fx)
        mul!(g₍ₓ₎,opt.∇fx,opt.Δx)
        g₍ₓ₎.+=opt.fx
        g₍ₓ₎ .+= 0.5 .* (opt.ρ) .* sum(abs2,(opt.Δx)./(opt.σ))
        println("   inner Current f(x+Δx): $(opt.f_and_∇f(opt.x+opt.Δx)[1])")
        println("   inner Current g(x): $(g₍ₓ₎)")
        #println("       g(x)inner Current opt.fx: $(opt.fx)")
        #println("       g(x)inner Current opt.∇fx: $(opt.∇fx)")
        #println("       g(x)inner Current opt.∇fx*opt.Δx: $(opt.∇fx*opt.Δx)")
        #println("       g(x)inner Current 2-defree: $(0.5 .* (opt.ρ) .* sum(abs2,(opt.Δx)./(opt.σ)))")
        println("   inner Current ρ: $(opt.ρ)")
        println("   inner Current σ: $(opt.σ)")
        println("   ####################################")
        println("   ####################################")
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
        update =  sign.(opt.x - opt.x⁻¹).*sign.(opt.Δx)

        println("Current x: $(opt.x)")
        println("Current x+Δx: $(opt.x+opt.Δx)")
        println("Current update: $(update)")
        println("Current σ: $(opt.σ)")
        for j in 1:opt.n
            if update[j] == 1
                opt.σ[j] *= 2.0
            elseif update[j] == -1
                opt.σ[j] *= 0.5
            end
        end 
        opt.ρ .*= 0.5
        opt.x⁻¹ .= opt.x
        opt.x .= opt.x .+ opt.Δx
        println("Current σ after update: $(opt.σ)")
        if norm(opt.Δx) < opt.xtol_rel
            break
        end
    end
end
