using SafeTestsets
using Test
import Random

Random.seed!(1234)

@safetestset "Dual toy problem" begin include("test_dual.jl") end
@safetestset "Rosenbrock" include("test_rosenbrock.jl") end