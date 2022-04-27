#include("../src/SparseCCSA.jl")
include("../src/ccsa.jl")
include("../src/algo.jl")
#include("../src/model.jl")
using Test
using LinearAlgebra
using CatViews

##### Formal test #####
#######################
#######################
function f(x)
    return [sum(abs2,x),x[1]-1.0]
end
function ∇f(x) #(m+1)*n
    return [2x[1];1.0;;]
end
function f_and_∇f(x)
    return f(x),∇f(x)
end
# min x^2 s.t. x-1<0
n=1
m=1
ρ=[10.0,10.0] #m+1
σ=[10.0] #n
x=[10.0]
opt=CCSAState(n,m,f_and_∇f,ρ,σ,x)
optimize(opt)
opt.x


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


function f(x)
    return [sum(abs2,x)]
end
function ∇f(x) #(m+1)*n
    return [2x[1] 2x[2] 2x[3];]
end
function f_and_∇f(x)
    return f(x),∇f(x)
end
n=3
m=0
ρ0=[10.0] #m+1
σ0=10.0*ones(n) #n
x0=50.0*ones(n) #n
st=CCSAState(n,m,f_and_∇f,ρ0,σ0,x0)
optimize(st)
st.x