using ImplicitAdjoints
using Random
using Plots
using Symbolics
using SparseCCSA
using Statistics

n, p = 20, 40
S = 8
α = 0.1
β = 0
G = randn(n, p)
reg = L1(p)

u = zeros(p)
u[randperm(p)[1:S]] .= rand(S)
η = randn(n)
y = G * u
y += 0.05 * mean(abs.(y)) * η

# Solve problem with FISTA
uest, info = genlasso(G, y, α, β, 1000, 1e-12, reg) 
plt = scatter(uest, label="uest")

# Setup sparse Jacobian
u0 = rand(p)
t = 2 * abs.(u0) # start the t's with some slack
u_and_t = vcat(u0, t)

function f_and_∇f(fx, ∇fx, u_and_t)
    u = @view u_and_t[1:p]
    t = @view u_and_t[p+1:2p]
    fx[1] = sum((G * u - y).^2) + α * sum(t)
    fx[2:end] .= vcat(u - t, -u - t)
    ∇fx[1, :] .= vcat(2 * G' * (G * u - y), fill(α, p))
    return nothing
end

∇cons = Symbolics.jacobian_sparsity((y,x) -> (y .= vcat(x[1:p] - x[p+1:2p], x[1:p] + x[p+1:2p])), vcat(u-t,u+t), u_and_t)

∇fx = vcat(zeros(2*p)', ∇cons)
fx = zeros(2p + 1)

f_and_∇f(fx, ∇fx, u_and_t)
fx[1]
sum((G * u - y).^2) + α * sum(t)

lb = vcat(fill(-Inf, p), zeros(p))
ub = fill(Inf, 2p)

opt = init(f_and_∇f, lb, ub, 2p, 2p; x0=u_and_t, max_iters=2, max_inner_iters=10, max_dual_iters=5, max_dual_inner_iters=5, ∇fx_prototype=∇fx);
sol = solve!(opt)


usol = sol.x[1:p]
tsol = sol.x[p+1:2p]
maximum(usol - tsol)

