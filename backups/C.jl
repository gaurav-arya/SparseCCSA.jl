include("../src/ccsa.jl")
using LinearAlgebra
using CatViews
using BenchmarkTools
using Profile
using ProfileView
using PProf
a1 = 2;
b1 = 0;
a2 = -1;
b2 = 1;
function f(x)
    return [sqrt(x[2]), (a1 * x[1] + b1)^3 - x[2], (a2 * x[1] + b2)^3 - x[2]]
end
function jac(x) #(m+1)*n
    return [0 1 / 2/x[2]; 3*a1*(a1 * x[1] + b1)^2 -1; 3*a2*(a2 * x[1] + b2)^2 -1]
end
function f_and_jac(x)
    return f(x), jac(x)
end
n = 2
m = 2
σ = [30.0, 30.0] #n
ρ = [100.0, 100.0, 100.0]
x = [1.234, 5.678]
lb = [-Inf, 0.0]
opt = CCSAState(n, m, f_and_jac, ρ, σ, x)
opt.lb = lb
opt.xtol_rel = 1e-4
@time iters = optimize(opt)
ProfileView.@profview optimize(opt)
