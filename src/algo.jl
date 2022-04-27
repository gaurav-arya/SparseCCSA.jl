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
