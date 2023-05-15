module SetupLasso

using Random

# n, p = 20, 40
# 
# α = 0.001
# β = 0
# reg = L1(p)
function setup_lasso(n, p)
    S = 3
    G = randn(n, p)

    rng = Xoshiro(n + p) # make problem deterministic given n and p

    u = zeros(p)
    u[randperm(rng, p)[1:S]] .= rand(S)
    η = randn(rng, n)
    y = G * u
    y += 0.05 * mean(abs.(y)) * η

    return u, G, y, α, β
end