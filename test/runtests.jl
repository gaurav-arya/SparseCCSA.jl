using SparseCCSA
using Test

@testset "SparseCCSA.jl" begin
    # Write your tests here.
    function f(x)
        return sum(abs2,x)
    end
    function fgrad(x) #(m+1)*n
        return [2x[1];]
    end
    n=1
    m=0
    ρ0=[100.0,100.0]
    σ0=[10.0]
    x0=[50.0]
    CCSAState(n,m,f_fgrad,ρ0,σ0,x0)
end
