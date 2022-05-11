using SparseCCSA
using Test
using LinearAlgebra
@testset "SparseCCSA.jl" begin
    # Write your tests here.
    function test1(bound)
        function f_and_∇f(x)
            f=[abs2(x[1]),x[1]-bound]
            ∇f=[2x[1];1.0;;]
            return f,∇f
        end
        n=1
        m=1
        σ=[30.0] #n
        ρ=[2.0,0.0001]*σ[1]^2 #m+1
        x=[-10.0]
        lb=[-100.0]
        ub=[100.0]
        opt=CCSAState(n,m,f_and_∇f,x,ρ=ρ,σ=σ,lb=lb,ub=ub)
        optimize(opt)
        return opt.x
    end
    function test2()
        function f_and_grad(x)
            fx = [x[1]^2 * x[2], x[1] - 3, x[1] + 4, x[2] - 4]
            gradx = [2*x[1]* x[2] x[1]^2; 1 0; 1 0; 0 1]
            fx, gradx
        end
        n=2
        m=3
        ρ=[1000.,1000.,1000.,1000.] #m+1
        σ=[100.,100.] #n
        lb=[-100.0,-100.0]
        ub=[100.0,100.0]
        x=[0.0,0.0]
        xtol_rel=1e-6
        st = CCSAState(n,m,f_and_grad,x,ρ=ρ,σ=σ,lb=lb,ub=ub,xtol_rel=xtol_rel)
        optimize(st)
        return st.x
    end
    @test norm(test1(-1)-[-1.0])<0.001 # the constraint is tight
    @test norm(test1(1)-[0.])<0.001 # the constraint is tight
    @test norm(test2()-[-100.0, -100.0])<1.0 # the box constraint is tight
end
