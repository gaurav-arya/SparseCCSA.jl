using SparseCCSA
using SparseCCSA: inner_iterations
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
end

@testset "integration" begin
    function test1(bound)
        function f_and_∇f(x)
            f = [abs2(x[1]), x[1] - bound]
            ∇f = [2x[1]; 1.0]
            return f, ∇f
        end
        n = 1
        m = 1
        σ = [30.0] # n
        ρ = [2.0, 0.0001] * σ[1]^2 # m + 1
        x = [-10.0]
        lb = [-100.0]
        ub = [100.0]
        opt = CCSAState(n, m, f_and_∇f, ρ, σ, x, lb=lb, ub=ub)
        optimize(opt)
        return opt.x
    end
    @test test1(-1.0) ≈ [-1.0] atol = 1e-5
    @test test1(1.0) ≈ [0.0] atol = 1e-5

    function test2()
        function f_and_grad(x)
            fx = [
                x[1]^2 * x[2],
                x[1] - 3,
                x[1] + 4,
                x[2] - 4
            ]
            gradx = [
                2*x[1]*x[2] x[1]^2
                1.0 0.0
                1.0 0.0
                0.0 1.0
            ]
            fx, gradx
        end
        n = 2
        m = 3
        ρ = fill(1000.0, m + 1)
        σ = fill(100.0, n)
        lb = fill(-100.0, n)
        ub = fill(100.0, n)
        x = [-5.0, 0.0]
        st = CCSAState(n, m, f_and_grad, ρ, σ, x, lb=lb, ub=ub)
        optimize(st)
        return st.x
    end
    @test test2() ≈ [-100.0, -100.0] atol = 1e-5

    function test3()
        function f(x)
            return [sum(abs2, x), x[1] + 1.0]
        end
        function ∇f(x) #(m+1)*n
            return [2x[1]; 1.0]
        end
        function f_and_∇f(x)
            return f(x), ∇f(x)
        end
        n = 1
        m = 1
        σ = [30.0] #n
        ρ = [2.0, 0.0001] * σ[1]^2 #m+1
        x = [-10.0]
        opt = CCSAState(n, m, f_and_∇f, ρ, σ, x)
        optimize(opt)
        opt.x
    end
    @test test3() ≈ [-1.0] atol = 1e-5
end

@testset "inner_iterations" begin
    function f(x)
        return [sum(abs2, x), x[1] + 1.0]
    end
    function ∇f(x) # (m + 1) × n
        return [2x[1]; 1.0]
    end
    function f_and_∇f(x)
        return f(x), ∇f(x)
    end
    n = 1
    m = 1
    σ = [30.0] # n
    ρ = [2.0, 0.0001] * σ[1]^2 # m + 1
    x = [-10.0]
    opt = CCSAState(n, m, f_and_∇f, ρ, σ, x)
    inner_iterations(opt)
    @test opt.Δx ≈ [5.0] atol = 1e-5
end

@testset "no constraint" begin
    function fundamental_no_constraints_test(n)
        function f(x)
            return [sum(abs2, x)]
        end
        function ∇f(x) # (m + 1) × n
            return 2 * x'
        end
        function f_and_∇f(x)
            return f(x), ∇f(x)
        end
        m = 0
        ρ0 = [101.0] # m + 1
        σ0 = fill(0.1, n) # n
        x0 = fill(50.0, n) # n
        st = CCSAState(n, m, f_and_∇f, ρ0, σ0, x0)
        optimize(st)
        return st.x
    end
    @test fundamental_no_constraints_test(6) ≈ fill(0.0, 6) atol = 1e-4
end

@testset "NLopt.jl tutuorial" begin
    function myfunc(x::Vector, grad::AbstractVector)
        if length(grad) > 0
            grad[1] = 0
            grad[2] = 0.5 / sqrt(x[2])
        end
        sqrt(x[2])
    end

    function myconstraint(x::Vector, grad::AbstractVector, a, b)
        if length(grad) > 0
            grad[1] = 3a * (a * x[1] + b)^2
            grad[2] = -1
        end
        (a * x[1] + b)^3 - x[2]
    end

    n = 2
    m = 2
    function f_and_∇f(x)
        f = Vector{Float64}(undef, m + 1)
        ∇f = Matrix{Float64}(undef, m + 1, n)
        f[1] = myfunc(x, @view ∇f[1, :])
        f[2] = myconstraint(x, (@view ∇f[2, :]), 2, 0)
        f[3] = myconstraint(x, (@view ∇f[3, :]), -1, 1)
        f, ∇f
    end
    lower_bounds = [-Inf, 0.0]
    xtol_rel = 1e-4
    x₀ = [1.234, 5.678]
    ρ = ones(m + 1)
    σ = ones(n)
    opt = CCSAState(n, m, f_and_∇f, ρ, σ, x₀, lb=lower_bounds, xtol_rel=xtol_rel)
    optimize(opt)
    @test opt.x[1] ≈ 1 / 3 rtol = 1e-4
    @test opt.x[2] ≈ 8 / 27 rtol = 1e-3
end