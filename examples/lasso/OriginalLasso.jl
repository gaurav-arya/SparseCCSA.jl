using SparseCCSA
using LinearAlgebra
using CatViews
using BenchmarkTools
using SparseArrays 
using BenchmarkTools
using Plots
m = 0
α= 0.1
time=Array{Float64}(undef,10000)
iter=Array{Int}(undef,10000)
value=Array{Float64}(undef,10000)
function cb()
    value[opt.iters]=opt.fx[1]
    println(opt.iters)
end
n=1000
A=sparse(Matrix(SymTridiagonal(2*ones(n),ones(n))))
x_true= zeros(n)*10
x_true.= sprand(n,0.1)*10
y=A*x_true
function f_and_jac(x)
    f=[sum((A*x-y).^2)+α*norm(x,1)]
    jac=(A*x-y)'*A.+α.*signbit.(x)'
    return f,jac
end
opt = CCSAState(n, 0, f_and_jac,zeros(n),max_iters=2)
SparseCCSA.optimize(opt,callback=cb) 
plot(1:opt.iters,value[1:opt.iters],yscale=:log10)
