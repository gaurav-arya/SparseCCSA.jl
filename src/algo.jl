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
function g_dual(ρ,σ,f0,f_grad,y)
#f_grad is already a m+1*n matrix, f_grad=fgrad(x)
    #L needs y[1:m], g0,g1...gn
    #L needs y[1:m], ∇f0[1:n],∇f1...∇fn, ρ[0:m], σ[1:n]
    #m: constraints
    #y: lagrangian multiplyers
    #return δ
    n=length(σ)
    m=length(ρ)-1
    u=ρ[1]+dot(y, @view ρ[2:m+1])
    a=0.0
    b=0.0
    g_sum=f0
    for j in 1:n
        #solve gj respectively
        a=u/(2*σ[j]^2)#aⱼ
        b=f_grad[1,j]+dot(y, @view f_grad[2:m+1,1:n])#bⱼ
        δ=-b/(2a)
        #Recall gj=min_{|δj|<σj} {aδ²+bδ}
        #Min: δ=-b/2a
        if δ>σ[j]
            g_sum+=a*σ[j]^2+b*σ[j]
        elseif δ<-σ[j]
            g_sum+=a*σ[j]^2-b*σ[j]
        else 
            g_sum+=-b^2/(4a)
        end
    end
    return g_sum
end
function ggrad(ρ,σ,f_grad,y)
#f_grad is already a m+1*n matrix, f_grad=fgrad(x)
#output: 1:n, no constraint
    n=length(σ)
    m=length(ρ)-1
    g_grad=zeros(n)
    u=ρ[1]+dot(y, @view ρ[2:m+1])
    for j in 1:m
        a=u/(2*σ[j]^2)#aⱼ
        b=f_grad[1,j]+dot(y, @view f_grad[2:m+1,1:n])#bⱼ
        δ=-b/(2a)
        if δ>σ[j]
            δ=σ[j]
        elseif δ<-σ[j]
            δ=-σ[j]
        end
        g_grad.+=ρ./(2*σ[j]^2).*δ^2 .+ (@view f_grad[j,:]).*δ
    end
    return g_grad
end
function inner_iterations(opt::CCSAOpt, xˡ::AbstractVector)
    #∇f_xˡ = Array{Float64,2}(undef, opt.m + 1, opt.n)
    #f_xˡ = map(i -> opt.f[i](xₖ, @view ∇f_xᵏ[i, :]), 1:opt.m+1) # TODO: adjust
    f_grad=opt.fgrad(xˡ)
    f0=opt.f(xˡ)
    inner_nevals=0
    ρ_again=[1.0]
    σ_again=copy(opt.σ)
    while true
        inner_nevals+=1
        # Recursively call optimize with a new opt object
        # optimize g(y) using dual_func! 
        # once we find the best y, how do we find x^(k+1)?
        opt_again=CCSAOpt(opt.n,0,opt.lower_bounds,opt.upper_bounds,opt.xtol_rel,y->g_dual(opt.ρ,opt.σ,f0,f_grad,y),y->ggrad(opt.ρ,opt.σ,f_grad,y),ρ_again,σ_again)
        xˡ⁺¹ = optimize_simple(opt_again,xˡ) # TODO: optimize inner_iteration dual
        g_xˡ⁺¹ = Array{Float64}(undef,opt.n)
        for i in 1:m+1
            g_xˡ⁺¹[i]=dot(f_grad[i,:],xˡ⁺¹-xˡ)+ρ[i]^2/2*sum(abs2,(xˡ⁺¹-xˡ)./σ)
        end
        f_xˡ⁺¹ = map(fᵢ -> fᵢ(xˡ⁺¹, []), opt.f)
        conservative = g_xˡ⁺¹ .>= f_xˡ⁺¹
        if all(conservative)
            println("CCSA inner iteration: rho -> $ρ")
            println("CCSA inner iteration: loops -> $inner_nevals")
            println("CCSA inner iteration: xᵏˡ -> $xᵏ")
            break
        end
        opt.ρ[.!conservative] *= 2 #update ρ⁽ᵏˡ⁾
    end
    return xᵏ⁺¹
end
function optimize_simple(opt::CCSAOpt, x⁰::AbstractVector)
    xᵏ⁻¹ = copy(x⁰) 
    xᵏ = copy(x⁰)
    xᵏ⁺¹=copy(x⁰)
    ρ=opt.ρ
    σ=opt.σ
    a=0.0
    b=0.0
    g_sum=0.0
    f_grad=Array{Float64}(undef,1,n)
    while true
        f_grad.=opt.fgrad(xᵏ)
        f0=opt.f(xᵏ)
        while true
            g_sum=f0
            for j in 1:n
                a=ρ[1]/(2*σ[j]^2)#aⱼ
                b=f_grad[opt.m+1,j]#bⱼ
                δ=-b/(2a)
                if δ>σ[j]
                    g_sum+=a*σ[j]^2+b*σ[j]
                    xᵏ⁺¹[j]=xᵏ[j]+σ[j]
                elseif δ<-σ[j]
                    g_sum+=a*σ[j]^2-b*σ[j]
                    xᵏ⁺¹[j]=xᵏ[j]-σ[j]
                else 
                    g_sum+=-b^2/(4a)
                    xᵏ⁺¹[j]=xᵏ[j]+δ
                end
            end
            conservative = g_sum >= opt.f(xᵏ⁺¹)
            if (conservative)
                break
            end
            ρ[1] *= 2
        end
        if norm(xᵏ⁺¹ - xᵏ) < opt.xtol_rel
            break
        end
        opt.ρ .*= 0.5
        signᵏ = sign.(xᵏ - xᵏ⁻¹)
        signᵏ⁺¹ = sign.(xᵏ⁺¹ - xᵏ)
        update = signᵏ .* signᵏ⁺¹
        map(1:opt.n) do j
            if update[j] == 1
                opt.σ[j] *= 2.0
            elseif update[j] == -1
                opt.σ[j] *= 0.5
            end
        end #update σ
        xᵏ⁻¹ .= xᵏ
        xᵏ .= xᵏ⁺¹
    end
    return xᵏ
end
function optimize(opt::CCSAOpt, x⁰::AbstractVector)
    if opt.m==0
        return optimize_simple(opt, x⁰)
    end
    xᵏ⁻¹ = copy(x⁰) 
    xᵏ = copy(x⁰)
    outer_nevals=0
    while true
        outer_nevals+=1
        xᵏ⁺¹ = inner_iterations(opt, xᵏ)
        opt.ρ .*= 0.5
        signᵏ = sign.(xᵏ - xᵏ⁻¹)
        signᵏ⁺¹ = sign.(xᵏ⁺¹ - xᵏ)
        update = signᵏ .* signᵏ⁺¹
        map(1:opt.n) do j
            if update[j] == 1
                opt.σ[j] *= 2.0
            elseif update[j] == -1
                opt.σ[j] *= 0.5
            end
        end #update σ
        xᵏ⁻¹ .= xᵏ
        xᵏ .= xᵏ⁺¹
        if norm(xᵏ - xᵏ⁻¹) < opt.xtol_rel
            print("CCSA outer iteration: loops -> $outer_nevals")
            break
        end
    end
    return xᵏ
end
