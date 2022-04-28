using SparseCCSA
using Test

@testset "SparseCCSA.jl" begin
    # Write your tests here.
    function fundamental_no_constraints_test(n)
        function f(x)
            return [sum(abs2,x)]
        end
        function ∇f(x) #(m+1)*n
            return 2*x[:,:]'
        end
        function f_and_∇f(x)
            return f(x),∇f(x)
        end
        m=0
        ρ0=[10.0] #m+1
        σ0=10.0*ones(n) #n
        x0=50.0*ones(n) #n
        st=CCSAState(n,m,f_and_∇f,ρ0,σ0,x0)
        optimize(st)
        return st.x
    end
    @test norm(fundamental_no_constraints_test(5))<0.01
end
