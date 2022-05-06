using SparseCCSA
using Test
using LinearAlgebra
using SparseArrays

@testset "constructor" begin
    nvar = 30
    ncon = 40
    function test_constructor(T::DataType)
        function f_and_∇f(x)
            rand(T, ncon + 1, nvar) * x, rand(T, ncon + 1, nvar)
        end
        ρ = ones(T, nvar + 1)
        σ = ones(T, ncon)
        x₀ = zeros(T, nvar)
        lower_bound = -rand(T, nvar)
        upper_bound = rand(T, nvar)
        xtol_rel = T(1e-4)

        opt₁ = CCSAState(nvar, ncon, f_and_∇f, ρ, σ, x₀)
        @test typeof(opt₁) == CCSAState{T}
        @test opt₁.n == nvar
        @test opt₁.m == ncon
        @test opt₁.x == x₀
        @test opt₁.lb == fill(typemin(T), nvar)
        @test opt₁.ub == fill(typemax(T), nvar)
        @test opt₁.xtol_rel == T(1e-5)
        opt₂ = CCSAState(nvar, ncon, f_and_∇f, ρ, σ, x₀, lb=lower_bound)
        @test opt₂.lb == lower_bound
        @test opt₂.ub == fill(typemax(T), nvar)
        @test opt₂.xtol_rel == T(1e-5)
        opt₃ = CCSAState(nvar, ncon, f_and_∇f, ρ, σ, x₀, lb=lower_bound, ub=upper_bound)
        @test opt₃.lb == lower_bound
        @test opt₃.ub == upper_bound
        @test opt₃.xtol_rel == T(1e-5)
        opt₄ = CCSAState(nvar, ncon, f_and_∇f, ρ, σ, x₀, xtol_rel=xtol_rel)
        @test opt₄.lb == fill(typemin(T), nvar)
        @test opt₄.ub == fill(typemax(T), nvar)
        @test opt₄.xtol_rel == xtol_rel
    end
    test_constructor(Float16)
    test_constructor(Float32)
    test_constructor(Float64)
end

@testset "sparse Jacobian constuctor" begin
    nvar = 100
    ncon = 200
    function f_and_∇f(x)
        sprand(Float64, ncon + 1, nvar, 0.3) * x, sprand(Float64, ncon + 1, nvar, 0.2)
    end
    ρ = ones(nvar + 1)
    σ = ones(ncon)
    x₀ = zeros(nvar)
    sparse_opt = CCSAState(nvar, ncon, f_and_∇f, ρ, σ, x₀)
    # Dense Jacobian should not be allocated if sparsity is provided
    @test typeof(sparse_opt.∇fx) <: AbstractSparseMatrix
    @test typeof(sparse_opt.∇fx) <: DenseMatrix broken = true
end

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
        opt=CCSAState(n,m,f_and_∇f,ρ,σ,x,lb=lb,ub=ub)
        optimize(opt)
        return opt.x
    end
    function test2()
        function f_and_grad(x)
            fx = [x[1]^2 * x[2], x[1] - 3, x[1] + 4, x[2] - 4]
            gradx = [2*x[1] x[1]^2; 1 0; 1 0; 0 1]
            fx, gradx
        end
        n=2
        m=3
        ρ=[1000.,1000.,1000.,1000.] #m+1
        σ=[100.,100.] #n
        lb=[-100.0,-100.0]
        ub=[100.0,100.0]
        x=[0.0,0.0]
        st = CCSAState(n,m,f_and_grad,ρ,σ,x,lb=lb,ub=ub)
        optimize(st)
        return st.x
    end
    @test norm(test1(-1)-[-1.0 ])<0.001 # the constraint is tight
    @test norm(test1(1)-[0.])<0.001 # the constraint is tight
    @test norm(test2()-[-100.0, -100.0])<1.0 # the box constraint is tight
end
