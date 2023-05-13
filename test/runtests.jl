using SafeTestsets
using Test
import Random

Random.seed!(1234)

@safetestset "Dual toy problem" begin include("dual_toy_problem.jl") end
@safetestset "Rosenbrock" include("rosenbrock.jl") end