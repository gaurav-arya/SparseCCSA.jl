include("../examples/lasso/SetupLasso.jl")
include("../examples/lasso/FISTASolver.jl")
include("../examples/lasso/SparseCCSALassoData.jl")
using .SetupLasso
using .FISTASolver
using .SparseCCSALassoData
using LinearAlgebra
using Test

# Problem parameters

n = 4
p = 8
S = 2
noise_level = 0.01
α = 1e-1
β = 0.0

@testset "Convergence" begin
    for i in 1:5
        # Initialize problem
        (;u, G, y) = setup_lasso(n, p, S, noise_level)
        # Solve problem with FISTA
        uest, info = fista(G, y, α, β, 10000, 1e-16, L1(p))
        # Solve problem with CCSA
        h, opt = sparseccsa_lasso_data(G, y, α, β; xtol_rel=1e-9, dual_ftol_rel=1e-8)
        uestsp = h.x[end][1:p]
        @test norm(uestsp - uest) / norm(uest) < 1e-6 
    end
end