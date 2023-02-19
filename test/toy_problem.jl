@testset "Dual evaluator" begin
    #=
    The below corresponds to a a convexified problem with
    g0(δ) = 2 δ1 + δ2 + 1/2 (δ1^2 + δ2^2)
    g1(δ) = δ1 + 1/2 (δ1^2 + δ2^2)
    g2(δ) = δ2 + 1/2 (δ1^2 + δ2^2),

    giving dual objective h(λ) = - min_{|δ| .< 1} g0(δ) + λ1 g1(δ) + λ2 g2(δ),
    negated to make it a minimization problem.

    The solution is δ1 = -(2 + λ1) / (1 + λ1 + λ2) and δ2 = -(1 + λ2) / (1 + λ1 + λ2),
    subject to additional clamping.
    This may then be plugged into the objective.
    =#
    δ_solution(λ) = clamp.([-(2 + λ[1]), -(1 + λ[2])] / (1 + sum(λ)), -1, 1)
    g0(δ) = 2 * δ[1] + δ[2] + 1/2 * (δ[1]^2 + δ[2]^2)
    g1(δ) = δ[1] + 1/2 * (δ[1]^2 + δ[2]^2)
    g2(δ) = δ[2] + 1/2 * (δ[1]^2 + δ[2]^2)
    function h(λ, δ)
        return -(g0(δ) + λ[1] * g1(δ) + λ[2] * g2(δ))
    end
    h(λ) = h(λ, δ_solution(λ))
    δ_solution([0., 0.])

    δ_solution([0.0,0.0])


    # Form dual evaluator 
    # TODO: init_iterate is really mostly for initializing the type, abusing here to get the right values.
    iterate = SparseCCSA.init_iterate(; n = 2, m = 2, x0 = ones(2), jac_prototype = [2 1; 1 0; 0 1], lb=[-Inf, -Inf], ub=[Inf, Inf])    
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
                # Test that δi is clamped to -1 or 1, and that gradient is either 0 or agrees in sign with clamped value.
                @test (abs(δi) ≈ 1.0) && ((sign(δ_gradi) == sign(δi)) || (δ_gradi + 1 ≈ 1.0))
            end
        end

        gλ = zeros(1)
        ∇gλ = zeros(2)
        dual_evaluator(gλ, ∇gλ, λ)
        @test gλ[1] ≈ h(λ) # dual objective
        @test ∇gλ ≈ ForwardDiff.gradient(h, λ) # dual gradient 
        @test dual_evaluator.buffers.δ ≈ δ_solution(λ) # primal solution for dual = λ
    end

end