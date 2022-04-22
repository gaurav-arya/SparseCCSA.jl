#include("../src/SparseCCSA.jl")
include("../src/ccsa.jl")
include("../src/algo.jl")

lower_bounds=zeros(1)
upper_bounds=zeros(1)
xtol_rel=1e-5
################## Is optmize_simple WORK? ##########
n=1
m=0
function f(x)
    return sum(abs2,x)
end
function fgrad(x) #(m+1)*n
    return [2x[1];]
end
ρ=[100.0,100.0]
σ=[10.0]
x0=[50.0]
opt=CCSAOpt(1,0,lower_bounds,upper_bounds,xtol_rel,f,fgrad,ρ,σ)
x=optimize_simple(opt,x0)
x
# WE HOPE X=0
# YES!!!

################## Is optmize WORK? ##########
n=1
m=1 #x-1<0
function f(x)
    return [sum(abs2,x)]
end
function fgrad(x) #(m+1)*n
    A=Array{Float64}(undef,2,1)
    A[1,1]=2x[1]
    A[2,1]=1.0
    return A
end
ρ=[100.0,100.0]
σ=[10.0]
x0=[50.0]
opt=CCSAOpt(n,m,lower_bounds,upper_bounds,xtol_rel,f,fgrad,ρ,σ)
x=optimize(opt,x0)
x
# WE HOPE X=0

#=function f(a,b,c)
    d=a+b+c
    function g(a)
        return d
    end
    return g
end
g=f(1,2,3)
g(1)
g(2)
n=5
a=0
map(1:n) do j
    a*2
end

using SparseCCSA
using Test

@testset "SparseCCSA.jl" begin
    # Write your tests here.
    @test 1 == f(0) # will be removed later
end
=#
#=
struct F
    a
    b
end
a=F([1,1],2)
a.a[1]+=1
#WHY? a is a pointer
=#
