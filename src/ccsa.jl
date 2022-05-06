mutable struct CCSAState{T<:AbstractFloat}
    n::Integer # number of variables > 0
    m::Integer # number of inequality constraints ≥ 0
    lb::AbstractVector{T} # n lower bounds
    ub::AbstractVector{T} # n upper bounds
    f_and_∇f::Function # f(x) = (m+1, (m+1) x n linear operator)
    ρ::AbstractVector{T} # m + 1 penality weight
    σ::AbstractVector{T} # n radius of trust region
    x::AbstractVector{T} # current best feasible point
    xtol_rel::T # relative tolerence

    fx::AbstractVector{T} # (m+1) x 1 function values at x
    ∇fx::AbstractVecOrMat{T} # (m+1) x n linear operator of Jacobian at x
    a::AbstractVector{T} # n
    b::AbstractVector{T} # n
    Δx_zeroed::AbstractVector{T} # n
    Δx::AbstractVector{T} # n
    gλ::T
    ∇gλ::AbstractVector{T} # m Lagrange dual function gradient
    x⁻¹::AbstractVector{T}

    function CCSAState(
        n::Integer, # number of variables
        m::Integer, # number of inequality constraints
        f_and_∇f::Function,
        penality_weight::AbstractVector{T}, # (m + 1) penality weight ρ
        trust_radius::AbstractVector{T}, # n radius of trust region σ
        x₀::AbstractVector{T}; # initial feasible point
        lb::AbstractVector{T}=fill(typemin(T), n), # lower bounds, default -Inf
        ub::AbstractVector{T}=fill(typemax(T), n), # upper bounds, default Inf
        xtol_rel::T=T(1e-5) # relative tolerence
    ) where {T<:AbstractFloat}
        fx, ∇fx = f_and_∇f(x₀)
        new{T}(
            n,
            m,
            lb,
            ub,
            f_and_∇f,
            penality_weight,
            trust_radius,
            x₀,
            xtol_rel,
            fx,
            ∇fx,
            Vector{T}(undef, n),
            Vector{T}(undef, n),
            Vector{T}(undef, n),
            Vector{T}(undef, n),
            T(0),
            Vector{T}(undef, m),
            zeros(T, n)
        )
    end
end

# Returns the dual function g(λ) and ∇g(λ)
function dual_func!(λ::AbstractVector{T}, st::CCSAState{T}) where {T}
    λ_all = CatView([one(T)], λ)
    st.a .= dot(st.ρ, λ_all) ./ (2 .* st.σ .^ 2)
    mul!(st.b, st.∇fx', λ_all)
    @. st.Δx = clamp(-st.b / (2 * st.a), -st.σ, st.σ)
    @. st.Δx = clamp(st.Δx, st.lb - st.x, st.ub - st.x)
    st.gλ = dot(λ_all, st.fx) + sum(@. st.a * st.Δx^2 + st.b * st.Δx)
    mul!(st.∇gλ, st.∇fx[2:end, :], st.Δx)
    st.∇gλ += st.fx[2:end] + sum(abs2, st.Δx ./ st.σ) / 2 * st.ρ[2:end]
    return [st.gλ], st.∇gλ'
end

function optimize_simple(opt::CCSAState{T}) where {T}
    while true
        while true
            dual_func!(T[], opt)
            #println("           simple Current g(x): $(opt.gλ)")
            #println("           simple Current f(x+Δx): $(opt.f_and_∇f(opt.x+opt.Δx)[1][1][1])")
            if opt.gλ >= opt.f_and_∇f(opt.x + opt.Δx)[1][1]
                break
            end
            opt.ρ[1] *= 2
        end
        #println("       Simple Current x: $(opt.x)")
        #println("       Simple Current Δx: $(opt.Δx)")
        #println("       Simple Current ρ: $(opt.ρ)")
        #println("       Simple Current σ: $(opt.σ)")
        opt.ρ[1] *= 0.5
        update = sign.(opt.x - opt.x⁻¹) .* sign.(opt.Δx)
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
        opt.fx, opt.∇fx = opt.f_and_∇f(opt.x)
    end
    return nothing
end

function inner_iterations(opt::CCSAState{T}) where {T}
    # input: opt::CCSAState
    # output: update opt.Δx
    # opt.Δx is the solution of the dual problem
    # i.e. max{min{g0(x)+λ1g1(x)...}}=max{g(λ)}
    # gi are constructed by opt.ρ/opt.σ
    ρ_again = [one(T)]
    σ_again = ones(T, opt.m)
    while true
        opt.fx, opt.∇fx = opt.f_and_∇f(opt.x)
        max_problem(λ) = begin
            result = dual_func!(λ, opt)
            -result[1], -result[2]
        end
        opt_again = CCSAState( # dual problem
            opt.m, # number of variables
            0, # number of inequality constraints
            max_problem, # dual function and gradient
            ρ_again, # penality weight
            σ_again, # radius of trust regions
            ones(opt.m), # initial feasible point
            lb=zeros(opt.m) # lower bound for dual problem
        )
        optimize_simple(opt_again)
        dual_func!(opt_again.x, opt)
        println("   inner The optimal of dual λ: $(opt_again.x)")
        println("   inner The optimal of dual x: $(opt.x)")
        println("   inner The optimal of dual x+Δx: $(opt.x+opt.Δx)")
        g₍ₓ₎ = similar(opt.fx)
        mul!(g₍ₓ₎, opt.∇fx, opt.Δx)
        g₍ₓ₎ += opt.fx + opt.ρ * sum(abs2, opt.Δx ./ opt.σ) * 0.5
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
        conservative = (g₍ₓ₎ .>= opt.f_and_∇f(opt.x + opt.Δx)[1])
        if all(conservative)
            break
        end
        opt.ρ[.!conservative] *= 2
    end
end

function optimize(opt::CCSAState{T}) where {T}
    if opt.m == 0
        optimize_simple(opt)
        return
    end
    # test = dual_func!(zeros(T, opt.m), opt)
    while true
        inner_iterations(opt)
        update = sign.(opt.x - opt.x⁻¹) .* sign.(opt.Δx)

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
        opt.ρ *= 0.5
        opt.x⁻¹ .= opt.x
        opt.x += opt.Δx
        println("Current σ after update: $(opt.σ)")
        if norm(opt.Δx) < opt.xtol_rel
            break
        end
    end
end
