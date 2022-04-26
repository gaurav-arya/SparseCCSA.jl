#include("../src/SparseCCSA.jl")
include("../src/ccsa.jl")
include("../src/algo.jl")
include("../src/model.jl")
using Test
using LinearAlgebra
using CatViews
function f(x)
    return [sum(abs2,x)]
end
function ∇f(x) #(m+1)*n
    return [2x[1];;]
end
function f_and_∇f(x)
    return [f(x),∇f(x)]
end
n=1
m=0
ρ0=[100.0] #n
σ0=[10.0] #m+1
x0=[50.0]
opt=CCSAState(n,m,f_and_∇f,ρ0,σ0,x0)
##### Verify if dual_func!(λ::AbstractVector{T}, st::CCSAState) work #####
λ=Float64[]
dual_func!(λ, opt)
opt.gλ
opt.∇gλ
optimize_simple(opt)
opt.x
##### Formal test #####
#######################
#######################
function f(x)
    return [sum(abs2,x),x-1]
end
function ∇f(x) #(m+1)*n
    return [2x[1];1.0]
end
function f_and_∇f(x)
    return [f(x),∇f(x)]
end
n=1
m=1
ρ=[100.0] #n
σ=[10.0,10.0] #m+1
x0=[50.0]
opt=CCSAState(n,m,f_and_∇f,ρ0,σ0,x0)
optimize(opt)
opt.x