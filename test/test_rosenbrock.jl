include("../examples/rosenbrock/src/RosenbrockExample.jl")
using .RosenbrockExample
using Test

df_sp = sparseccsa_dataframe(400)
df_n = nlopt_dataframe(334)

# Check that first 20 iterations match
@testset "Consistency with nlopt" begin
    @test all(isapprox(df_n.x[i], df_sp.x[i]; rtol=1e-3) for i in 1:20)
    @test all(isapprox(df_n.ρ[i], df_sp.ρ[i]; rtol=1e-3) for i in 1:20)
    @test all(isapprox(df_n.σ[i], df_sp.σ[i]; rtol=1e-3) for i in 2:20) # don't include i = 1 since no data for nlopt
end

@testset "Convergence" begin
    @test all(isapprox(df_n.x[end], df_sp.x[end]; rtol=1e-5))
end