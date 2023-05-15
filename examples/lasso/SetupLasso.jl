module SetupLasso
__revise_mode__ = :eval

using Random
using Statistics
using Symbolics
using SparseArrays

function setup_lasso(n, p, S)
    G = randn(n, p)

    rng = Xoshiro(n + p) # make problem deterministic given n and p

    u = zeros(p)
    u[randperm(rng, p)[1:S]] .= rand(S)
    η = randn(rng, n)
    y = G * u
    y += 0.05 * mean(abs.(y)) * η

    return (;u, G, y)
end

# TODO: support β
"""
Return the f_and_jac function for Lasso's epigraph formulation.
"""
function lasso_epigraph(G, y, α)
    n, p = size(G)
    ∇cons = spzeros(2p, 2p) 
    for i in 1:p
        ∇cons[i, i] = 1
        ∇cons[i, p + i] = -1
        ∇cons[i + p, i] = -1
        ∇cons[i + p, i + p] = -1
    end
    f_and_jac = (fx, jac, u_and_t) -> begin
        u = @view u_and_t[1:p]
        t = @view u_and_t[(p + 1):(2p)]
        fx[1] = sum((G * u - y) .^ 2) + α * sum(t)
        fx[2:end] .= vcat(u - t, -u - t)
        if jac !== nothing
            jac[1, :] .= vcat(2 * G' * (G * u - y), fill(α, p))
            jac[2:end, :] .= ∇cons
        end
        return nothing
    end
    return (;f_and_jac, jac_prototype = vcat(ones(2 * p)', ∇cons))
end

export setup_lasso, lasso_epigraph

end