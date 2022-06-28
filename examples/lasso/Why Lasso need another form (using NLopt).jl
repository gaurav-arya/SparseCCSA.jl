using NLopt
using BenchmarkTools
using LinearAlgebra
using SparseArrays
using Plots
n=10
A=SymTridiagonal(2*ones(n),ones(n))
A=rand(n,n)
β=1.0
x_true=ones(n)*10
x_true.= sprand(n,1,0.1)*10
y=A*x_true
function original()
    function myfunc(x::Vector, grad::Vector)
        grad.=A'*(A*x-y) .+ β.*signbit.(x)
        return sum((A*x-y).^2)+β*norm(x,1)
    end
    opt = Opt(:LD_CCSAQ, n)
    opt.maxtime=5
    opt.min_objective = myfunc
    ppp=Float64[]
    for i in 1:200
        opt.maxeval=i
        @time (minf,minx,ret) = NLopt.optimize(opt, zeros(n))
        append!(ppp,minf)
    end
    numevals = opt.numevals # the number of function evaluations
    #println("got $minf in $(opt.maxtime) seconds")
    p=plot(1:length(ppp),ppp,yscale=:log10,ylim=(1,1e4))
    display(p)
end
function replaced()
    function myfunc(x::Vector, grad::Vector)
        grad[1:n] .= A'*(A*(@view x[1:n])-y) 
        grad[n+1:2n] .=  β
        return sum((A*(@view x[1:n])-y).^2)+β*sum(@view x[n+1:2n])
    end
    function myconstraint_pos!(x::Vector, grad::Vector,i)
        grad.=0.0
        grad[i]=-1.0
        grad[i+n]=-1.0
        return -x[i]-x[i+n]
    end
    function myconstraint_neg!(x::Vector, grad::Vector,i)
        grad.=0.0
        grad[i]=1.0
        grad[i+n]=-1.0
        return x[i]-x[i+n]
    end
    opt = Opt(:LD_CCSAQ, 2n)
    opt.maxtime=5
    opt.min_objective = myfunc
    for i in 1:n
        inequality_constraint!(opt, (x,g)->myconstraint_pos!(x,g,i))
        inequality_constraint!(opt, (x,g)->myconstraint_neg!(x,g,i))
    end
    pp=Float64[]
    for i in 1:200
        opt.maxeval=i
        @time (minf,minx,ret) = NLopt.optimize(opt, zeros(2n))
        println(i)
        append!(pp,minf)
    end
    numevals = opt.numevals # the number of function evaluations
    #println("got $minf in $(opt.maxtime) seconds")
    p=plot(1:length(pp),pp,yscale=:log10,ylim=(1,1e4))
    display(p)
end
original()
replaced()
k=100
p=plot(1:k,[pp[1:k,:] ppp[1:k,:]],yscale=:log10,ylim=(3,1e3),label=["Replaced" "Original"],xlabel="iterations",ylabel="objective function value")