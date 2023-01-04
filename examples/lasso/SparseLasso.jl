using SparseCCSA
using LinearAlgebra
using SparseArrays 
using BenchmarkTools
using Profile

function make_problem(α, n) 
    A=sparse(Matrix(SymTridiagonal(2*ones(n),ones(n))))
    x_true= ones(n)*10; x_true.= sprand(n,0.1)*10
    y=A*x_true
    @btime mul!(y,A,x_true)
    I=Vector{Int64}(undef,6n)
    J=Vector{Int64}(undef,6n)
    V=Vector{Float64}(undef,6n)
    for i in 1:2n
        I[i]=1;J[i]=i; V[i]=α;
    end
    for i in 2n+1:3n
        I[i]=i-(2n-1); J[i]=i-(2n); V[i]=-1;
    end
    for i in 3n+1:4n
        I[i]=i-(3n-1); J[i]=i-(2n); V[i]=-1;
    end
    for i in 4n+1:5n
        I[i]=i-(3n-1); J[i]=i-(4n); V[i]=1;
    end
    for i in 5n+1:6n
        I[i]=i-(4n-1); J[i]=i-(4n); V[i]=-1;
    end
    ∇f=sparse(I,J,V)
    function creat_∇f(x)
        ∇f.nzval[1:3:3n-2].=A'*(A*(@view x[1:n])-y)
        return ∇f
    end
    f=Vector{Float64}(undef,2n+1)
    function f_and_∇f(x)
        f = zeros(2n+1)
        f[1]=sum((A*(@view x[1:n])-y).^2)+α*sum(@view x[n+1:2n])
        for i in 1:n
            f[i+1]=-x[i]-x[i+n]
            f[i+n+1]=x[i]-x[i+n]
        end
        return f,creat_∇f(x)
    end
    max_iters=3
    state = CCSAState(2n, 2n, f_and_∇f,zeros(2n),max_iters=max_iters,xtol_rel=1e-4)
    return state
end

α= 0.1
n= 100
state = make_problem(α, n)

function solve_problem(opt, n)
    value=Array{Float64}(undef,1000)
    recode_xi_stable=Array{Float64}(undef,2n,100)

    function cb()
        value[opt.iters]=opt.fx[1]
        println(opt.iters)
        recode_xi_stable[:,opt.iters].=opt.x
    end

    optimize(opt,callback=cb) 
end

opt = solve_problem(opt, n)

using Plots
plot(1:opt.iters,value[1:opt.iters],yscale=:log10,ylim=[100,1e3])
using CSV, Tables
CSV.write( open("xi_stable.csv", "a+"), Tables.table(recode_xi_stable); delim = ',')        
opt.max_iters+=1
Profile.print()
