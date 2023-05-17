using SafeTestsets
using Test
import Random

Random.seed!(1234)

@safetestset "Dual toy problem" begin include("test_dual.jl") end
@safetestset "Rosenbrock" begin include("test_rosenbrock.jl") end
@safetestset "LASSO" begin include("test_lasso.jl") end