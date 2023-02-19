using SparseCCSA
using Test
using LinearAlgebra
using SparseArrays
using ForwardDiff

@testset "Dual evaluator" begin
    #=
    The below corresponds to a a convexified problem with
    g0(δ) = 2 δ1 + δ2 + 1/2 (δ1^2 + δ2^2)
    g1(δ) = δ1 + 1/2 (δ1^2 + δ2^2)
    g2(δ) = δ2 + 1/2 (δ1^2 + δ2^2),

    giving dual objective h(λ) = min_{|δ| .< 1} g0(δ) + λ1 g1(δ) + λ2 g2(δ).

    The solution is δ1 = -(2 + λ1) / (1 + λ1 + λ2) and δ2 = -(1 + λ2) / (1 + λ1 + λ2),
    subject to additional clamping.
    This may then be plugged into the objective.
    =#
    δ_solution(λ) = clamp.([-(2 + λ[1]), -(1 + λ[2])] / (1 + sum(λ)), -1, 1)
    g0(δ) = 2 * δ[1] + δ[2] + 1/2 * (δ[1]^2 + δ[2]^2)
    g1(δ) = δ[1] + 1/2 * (δ[1]^2 + δ[2]^2)
    g2(δ) = δ[2] + 1/2 * (δ[1]^2 + δ[2]^2)
    function h(λ, δ)
        return g0(δ) + λ[1] * g1(δ) + λ[2] * g2(δ)
    end
    h(λ) = h(λ, δ_solution(λ))

    # Form dual evaluator 
    # TODO: init_iterate is really mostly for initializing the type, abusing here to get the right values.
    iterate = SparseCCSA.init_iterate(; n = 2, m = 2, x0 = ones(2), ∇fx_prototype = [2 1; 1 0; 0 1], lb=[-Inf, -Inf], ub=[Inf, Inf])    
    buffers = SparseCCSA.init_buffers(; T=Float64, n=2)
    dual_evaluator = SparseCCSA.DualEvaluator(; iterate, buffers)

    # Compare dual evaluator output to analytic
    for λ in ([1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [5.5, 3.2], rand(2))
        # Verify correctness of analytc solution.
        analytic_δgrad = ForwardDiff.gradient(δ -> h(λ, δ), δ_solution(λ)) 
        map(δ_solution(λ), analytic_δgrad, 1:2) do δi, δ_gradi, i
            if abs(δi) < 1
                # Add unit scale to both sides when comparing to 0 (see julia/#32244)
                @test δ_gradi + 1 ≈ 1.0
            else
                @test (abs(δi) ≈ 1.0) && (sign(δ_gradi) != sign(δi))
            end
        end

        dual_evaluator(gλ, ∇gλ, λ)
        @test gλ[1] ≈ h(λ) # dual objective
        @test ∇gλ ≈ ForwardDiff.gradient(h, λ) # dual gradient 
        @test dual_evaluator.buffers.δ ≈ δ_solution(λ) # primal solution for dual = λ
    end

end

##### Test the constructor #####
@testset "Constructor" begin
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

        opt₁ = CCSAState(nvar, ncon, f_and_∇f, x₀)
        @test typeof(opt₁) == CCSAState{T}
        @test opt₁.n == nvar
        @test opt₁.m == ncon
        @test opt₁.x == x₀
        @test opt₁.lb == fill(typemin(T), nvar)
        @test opt₁.ub == fill(typemax(T), nvar)
        @test opt₁.xtol_rel == T(1e-5)
        opt₂ = CCSAState(nvar, ncon, f_and_∇f, x₀, ρ = ρ, σ = σ, lb = lower_bound)
        @test opt₂.lb == lower_bound
        @test opt₂.ub == fill(typemax(T), nvar)
        @test opt₂.xtol_rel == T(1e-5)
        opt₃ = CCSAState(nvar, ncon, f_and_∇f, x₀, ρ = ρ, σ = σ, lb = lower_bound,
                         ub = upper_bound)
        @test opt₃.lb == lower_bound
        @test opt₃.ub == upper_bound
        @test opt₃.xtol_rel == T(1e-5)
        opt₄ = CCSAState(nvar, ncon, f_and_∇f, x₀, ρ = ρ, σ = σ, xtol_rel = xtol_rel)
        @test opt₄.lb == fill(typemin(T), nvar)
        @test opt₄.ub == fill(typemax(T), nvar)
        @test opt₄.xtol_rel == xtol_rel
    end
    test_constructor(Float16)
    test_constructor(Float32)
    test_constructor(Float64)
end

##### Test sparse matrix #####
@testset "Sparse Jacobian" begin
    nvar = 100
    ncon = 200
    function f_and_∇f(x)
        sprand(Float64, ncon + 1, nvar, 0.3) * x, sprand(Float64, ncon + 1, nvar, 0.2)
    end
    ρ = ones(nvar + 1)
    σ = ones(ncon)
    x₀ = zeros(nvar)
    sparse_opt = CCSAState(nvar, ncon, f_and_∇f, x₀, ρ = ρ, σ = σ)
    # Dense Jacobian should not be allocated if sparsity is provided
    @test typeof(sparse_opt.∇fx) <: AbstractSparseMatrix
end

##### No constraint #####
@testset "No constraint" begin
    ##### min x₁²+x₂²+x₃²+x₄²+... ##### 
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
        st = CCSAState(n, m, f_and_∇f, x0, ρ = ρ0, σ = σ0)
        optimize(st)
        return st.x
    end
    @test fundamental_no_constraints_test(6)≈fill(0.0, 6) atol=1e-4
end

##### Test a toy problem #####
@testset "test 1" begin
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
        opt = CCSAState(n, m, f_and_∇f, x, ρ = ρ, σ = σ, lb = lb, ub = ub)
        optimize(opt)
        return opt.x
    end
    @test test1(-1.0)≈[-1.0] atol=1e-5
    @test test1(1.0)≈[0.0] atol=1e-5
    @test norm(test1(-1) - [-1.0]) < 0.001 # the constraint is tight
    @test norm(test1(1) - [0.0]) < 0.001 # the constraint is tight
end

@testset "test 2" begin
    function test2()
        ##### min  x₁²x₂ #####
        ##### s.t. x₁ - 3 < 0 #####
        ##### s.t. x₁ + 4 < 0 #####
        ##### s.t. x₂ - 4 < 0 #####
        function f_and_grad(x)
            fx = [x[1]^2 * x[2], x[1] - 3, x[1] + 4, x[2] - 4]
            gradx = [2*x[1]*x[2] x[1]^2; 1 0; 1 0; 0 1]
            fx, gradx
        end
        n = 2
        m = 3
        ρ = [1000.0, 1000.0, 1000.0, 1000.0] #m+1
        σ = [100.0, 100.0] #n
        lb = [-100.0, -100.0]
        ub = [100.0, 100.0]
        x = [-10.0, -10.0]
        xtol_rel = 1e-6
        st = CCSAState(n, m, f_and_grad, x, ρ = ρ, σ = σ, lb = lb, ub = ub,
                       xtol_rel = xtol_rel)
        optimize(st)
        return st.x
    end
    @test norm(test2() - [-100.0, -100.0]) < 1.0 # the box constraint is tight
end
