function optimize_simple(opt::CCSAState)
    while true
        while opt.gλ < opt.f_and_∇f(opt.x+opt.Δx)[1][1]
            dual_func!(Float64[], opt)
            opt.ρ[1] *= 2
        end
        opt.ρ .*= 0.5
        update =  sign.(opt.x .- opt.x⁻¹).*sign.(opt.Δx)
        map(1:opt.n) do j
            if update[j] == 1
                opt.σ[j] *= 2.0
            elseif update[j] == -1
                opt.σ[j] *= 0.5
            end
        end 
        opt.x⁻¹ .= opt.x
        opt.x .= opt.x .+ opt.Δx
        if norm(opt.Δx) < opt.xtol_rel
            break
        end
        println("Suppose here is a callback")
        println("Current x: $(opt.x)")
    end
    nothing
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
        dual_func!(λ,opt)
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
        return nothing
    end
    dual_func!(zeros(opt.m),opt)
    while true
        inner_iterations(opt)
        opt.ρ .*= 0.5
        update = sign.(xᵏ .- x⁻¹) .* sign.(xᵏ⁺¹ .- xᵏ)
        
        map(1:opt.n) do j
            if update[j] == 1
                opt.σ[j] *= 2.0
            elseif update[j] == -1
                opt.σ[j] *= 0.5
            end
        end 
        opt.x⁻¹ .= opt.x
        opt.x .= opt.x .+ opt.Δx
        if norm(opt.Δx) < opt.xtol_rel
            break
        end
        println("Suppose here is a callback")
    end
end
