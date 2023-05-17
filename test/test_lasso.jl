include("../examples/lasso/SetupLasso.jl")
using .SetupLasso
using LinearAlgebra
using Test

## Initialize problem

n = 4
p = 8
S = 2
noise_level = 0.01
(;u, G, y) = setup_lasso(n, p, S, noise_level)
α = 1e-2
β = 0.0

## Solve problem with FISTA

include("../examples/lasso/FISTASolver.jl")
using .FISTASolver
uest, info = fista(G, y, α, β, 1000000, 1e-44, L1(p))
norm(uest - u) / norm(u)

## Solve problem with CCSA

include("../examples/lasso/SparseCCSALassoData.jl")
using .SparseCCSALassoData

h, opt = sparseccsa_lasso_data(G, y, α, β; xtol_rel=1e-11, dual_ftol_rel=1e-10)
ih = h.inner_history[1]
uestsp = h.x[end][1:p]

@testset "Convergence" begin
    @test norm(uestsp - uest) / norm(uest) < 1e-9 
end