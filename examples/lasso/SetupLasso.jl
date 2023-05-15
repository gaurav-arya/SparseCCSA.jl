module SetupLasso
__revise_mode__ = :eval

using Random
using Statistics
using Symbolics

function setup_lasso(n, p)
    S = 3
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
    ∇cons = Symbolics.jacobian_sparsity((y, x) -> (y .= vcat(x[1:p] - x[(p + 1):(2p)],
                                                            x[1:p] + x[(p + 1):(2p)])),
                                        zeros(2p), zeros(2p))
    f_and_jac = (fx, jac, u_and_t) -> begin
        u = @view u_and_t[1:p]
        t = @view u_and_t[(p + 1):(2p)]
        fx[1] = sum((G * u - y) .^ 2) + α * sum(t)
        fx[2:end] .= vcat(u - t, -u - t)
        if jac !== nothing
            jac[1, :] .= vcat(2 * G' * (G * u - y), fill(α, p))
            jac[2, :] .= ∇cons
        end
        return nothing
    end
    return (;f_and_jac, jac_prototype = vcat(zeros(2 * p)', ∇cons))
end

export setup_lasso, lasso_epigraph

end