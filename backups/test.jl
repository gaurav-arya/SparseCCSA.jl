include("../src/ccsa.jl")
using LinearAlgebra
using CatViews
using BenchmarkTools
using Profile
#using ProfileView
#using PProf
using SparseArrays
using BenchmarkTools
ncon = 0
#A=sprand(Float64, nvar, nvar, 0.3)
β = 0.1
#x_true=sprand(Float64, nvar, 1, 0.3)
time = Array{Float64}(undef, 10000)
pp = []
iter = Array{Int}(undef, 10000)
for n in 5000:100:5000
    A = Matrix(SymTridiagonal(2 * ones(n), ones(n)))
    #A=SymTridiagonal(2*ones(n),ones(n))
    x_true = ones(n) * 10
    y = A * x_true
    function f_and_jac(x)
        f = [sum((A * x - y) .^ 2) + β * norm(x, 1)]
        jac = (A * x - y)' * A .+ β .* signbit.(x)'
        return f, jac
    end
    sparse_opt = CCSAState(n, ncon, f_and_jac, xtol_rel = 10e-4)
    time[n] = @belapsed begin
        sparse_opt = CCSAState($n, 0, $f_and_jac)
        optimize(sparse_opt)
    end samples=5 evals=1
    #pp=optimize(sparse_opt)
    println(n)
    println(time[n])
end
plot(1:length(pp), pp, yscale = :log10, xlim = (-10, 550), ylim = (0.5e3, 1e7),
     label = "CCSA", xlabel = "iterations", ylabel = "objective function value")
using Plots
plot(250:1000, time[250:1000], xlim = (150, 1100), ylim = (0.00005, 0.0002),
     label = "time-dim graph")
plot(100:100:1000, time[100:100:1000], label = "time-dim graph")
plot(1:1000, iter[1:1000])

time_cun
time_jiu
@belapsed begin 1 == 1 end samples=100 evals=1
# Dense Jacobian should not be allocated if sparsity is provided

#
#=
a1=2;b1=0;a2=-1;b2=1;
function f(x)
    return [sqrt(x[2]),(a1*x[1]+b1)^3-x[2],(a2*x[1]+b2)^3-x[2]]
end
function jac(x) #(m+1)*n
    return [0 1/2/x[2]; 3*a1*(a1*x[1]+b1)^2 -1; 3*a2*(a2*x[1]+b2)^2 -1]
end
function f_and_jac(x)
    return f(x),jac(x)
end
n=2
m=2
σ=[30.0, 30.0] #n
ρ=[100.0, 100.0, 100.0]
x=[1.234, 5.678]
lb=[-Inf, 0.]
opt=CCSAState(n,m,f_and_jac,ρ,σ,x)
opt.lb=lb
opt.xtol_rel = 1e-4
@time optimize(opt)
ProfileView.@profview optimize(opt)
=#
