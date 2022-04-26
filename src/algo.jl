function optimize_simple(opt::CCSAState)
    while norm(opt.Δx) > opt.xtol_rel
        while opt.gλ < opt.f_and_∇f(opt.x+opt.Δx)[1][1]
            dual_func!(Float64[], opt)
            opt.ρ[1] *= 2
        end
        opt.ρ .*= 0.5
        update = (opt.x[1]#=ᵏ=# - opt.xᵏ⁻¹[1])*(opt.Δx[1]) 
        if update > 0
            opt.σ[1] *= 2.0
        elseif update < 0
            opt.σ[1] *= 0.5
        end
        opt.xᵏ⁻¹ .= opt.x
        opt.x .= opt.x .+ opt.Δx
        println("Suppose here is a callback")
    end
    nothing
end
function inner_iterations(opt::CCSAState)
    ρ_again=[1.0]
    σ_again=copy(opt.σ)
    while true
        opt_again=CCSAState(opt.n,0,λ->dual_func!(λ,opt),ρ_again,σ_again,zeros(opt.n),zeros(opt.n))
        
        optimize_simple(opt_again) 
        # NOW, FIND X(λ=opt_again.x) #
        # ALREADY IN opt.Δx IS IT ? #
        gxˡ⁺¹= Array{Float64}(undef,opt.m+1)
        mul!(gxˡ⁺¹,f_grad,opt.Δx)
        gxˡ⁺¹ .+= 0.5 .* (opt.ρ).^2 .* sum(abs2,(opt.Δx)./(opt.σ))
        conservative = ( gxˡ⁺¹ .>= opt.fx )
        if all(conservative)
            break
        end
        opt.ρ[.!conservative] *= 2
    end
end
function optimize(opt::CCSAState)
    if opt.m==0
        optimize_simple(opt)
    end
    while norm(opt.Δx) > opt.xtol_rel
        inner_iterations(opt)
        opt.ρ .*= 0.5
        update = (opt.x[1]#=ᵏ=# - opt.xᵏ⁻¹[1])*(opt.Δx[1]) 
        if update > 0
            opt.σ[1] *= 2.0
        elseif update < 0
            opt.σ[1] *= 0.5
        end
        opt.xᵏ⁻¹ .= opt.x
        opt.x .= opt.x .+ opt.Δx
        println("Suppose here is a callback")
    end
end
