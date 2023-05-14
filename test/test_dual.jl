using Test
using SparseCCSA
using ForwardDiff

@testset "Dual evaluator" begin
    #=
    The below corresponds to a a convexified problem with
    g0(Δx) = 2 Δx1 + Δx2 + 1/2 (Δx1^2 + Δx2^2)
    g1(Δx) = Δx1 + 1/2 (Δx1^2 + Δx2^2)
    g2(Δx) = Δx2 + 1/2 (Δx1^2 + Δx2^2),

    giving dual objective h(λ) = - min_{|Δx| .< 1} g0(Δx) + λ1 g1(Δx) + λ2 g2(Δx),
    negated to make it a minimization problem.

    The solution is Δx1 = -(2 + λ1) / (1 + λ1 + λ2) and Δx2 = -(1 + λ2) / (1 + λ1 + λ2),
    subject to additional clamping.
    This may then be plugged into the objective.
    =#
    Δx_solution(λ) = clamp.([-(2 + λ[1]), -(1 + λ[2])] / (1 + sum(λ)), -1, 1)
    g0(Δx) = 2 * Δx[1] + Δx[2] + 1 / 2 * (Δx[1]^2 + Δx[2]^2)
    g1(Δx) = Δx[1] + 1 / 2 * (Δx[1]^2 + Δx[2]^2)
    g2(Δx) = Δx[2] + 1 / 2 * (Δx[1]^2 + Δx[2]^2)
    function h(λ, Δx)
        return -(g0(Δx) + λ[1] * g1(Δx) + λ[2] * g2(Δx))
    end
    h(λ) = h(λ, Δx_solution(λ))
    Δx_solution([0.0, 0.0])

    Δx_solution([0.0, 0.0])

    # Form dual evaluator 
    cache = SparseCCSA.allocate_cache(; n = 2, m = 2, T = Float64, jac_prototype = [2 1; 1 0; 0 1])
    cache.x .= 1.0
    cache.ρ .= 1.0
    cache.σ .= 1.0
    cache.lb .= [-Inf, -Inf]
    cache.ub .= [Inf, Inf]
    dual_evaluator = SparseCCSA.DualEvaluator(; cache)

    # Compare dual evaluator output to analytic
    for λ in ([1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [5.5, 3.2], rand(2))
        # Verify correctness of analytc solution.
        analytic_Δxgrad = ForwardDiff.gradient(Δx -> h(λ, Δx), Δx_solution(λ))
        map(Δx_solution(λ), analytic_Δxgrad, 1:2) do Δxi, Δx_gradi, i
            if abs(Δxi) < 1
                # Add unit scale to both sides when comparing to 0 (see julia/#32244)
                @test Δx_gradi + 1 ≈ 1.0
            else
                # Test that Δxi is clamped to -1 or 1, and that gradient is either 0 or agrees in sign with clamped value.
                @test (abs(Δxi) ≈ 1.0) &&
                      ((sign(Δx_gradi) == sign(Δxi)) || (Δx_gradi + 1 ≈ 1.0))
            end
        end

        gλ = zeros(1)
        ∇gλ = zeros(2)'
        dual_evaluator(gλ, ∇gλ, λ)
        @test gλ[1] ≈ h(λ) # dual objective
        @test ∇gλ ≈ ForwardDiff.gradient(h, λ)' # dual gradient 
        @test cache.Δx ≈ Δx_solution(λ) # primal solution for dual = λ
    end
end
