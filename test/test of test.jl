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

# min x^2 s.t. x-1<0
function f(x)
    return [sum(abs2,x),x[1]-1.0]
end
function ∇f(x) #(m+1)*n
    return [2x[1];1.0;;]
end
function f_and_∇f(x)
    return f(x),∇f(x)
end
n=1
m=1
ρ=[10.0,10.0] #m+1
σ=[10.0] #n
x=[-10.0]
opt=CCSAState(n,m,f_and_∇f,ρ,σ,x)
optimize(opt)
opt.x

############## 下面是无约束测试 ##########
function fundamental_no_constraints_test()
    function f(x)
        return [sum(abs2,x)]
    end
    function ∇f(x) #(m+1)*n
        return 2*x[:,:]'
    end
    function f_and_∇f(x)
        return f(x),∇f(x)
    end
    n=1
    m=0
    ρ0=[10.0] #m+1
    σ0=10.0*ones(n) #n
    x0=50.0*ones(n) #n
    st=CCSAState(n,m,f_and_∇f,ρ0,σ0,x0)
    optimize(st)
    @assert norm(st.x)<0.0001
end
fundamental_no_constraints_test()
