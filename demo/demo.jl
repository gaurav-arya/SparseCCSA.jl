include("../src/model.jl")


n = 5
m = 3
x0 = fill(1.0, n)
lower = fill(-1e5, n)
upper = fill(1e6, n)
xtol_rel = 1e-6
ρ₀ = 1.0
ρᵢ = fill(1.0, m)
σ = fill(1.0, n)
model = CCSAModel(n, m, x0, lower, upper, xtol_rel, ρ₀, ρᵢ, σ)

function NLPModels.obj(model::CCSAModel, x::AbstractVector) # objective f₀
    @lencheck model.meta.nvar x
    increment!(model, :neval_obj)
    dot(rand(model.meta.nvar), x)
end
function NLPModels.grad(model::CCSAModel, x::AbstractVector) # gradient ∇f₀
    @lencheck model.meta.nvar x
    increment!(model, :neval_grad)
    rand(model.meta.nvar) .* x
end
function NLPModels.cons(model::CCSAModel, x::AbstractVector) # constraints fᵢ
    @lencheck model.meta.nvar x
    increment!(model, :neval_cons)
    rand(model.meta.ncon, model.meta.nvar) * x
end
function NLPModels.jac(model::CCSAModel, x::AbstractVector)
    # (sparse) Jacobian of constraints Jfᵢ
    @lencheck model.meta.nvar x
    increment!(model, :neval_jac)
    rand(model.meta.ncon, model.meta.nvar)
end
function NLPModels.jprod(model::CCSAModel, x::AbstractVector, v::AbstractVector)
    # product of constraints Jacobian and vector (Jfᵢ * v)
    @lencheck model.meta.nvar x v
    increment!(model, :neval_jprod)
    rand(model.meta.ncon)
end

r1 = NLPModels.obj(model, x0)
r2 = NLPModels.grad(model, x0)
r3 = NLPModels.cons(model, x0)
r4 = NLPModels.jac(model, x0)
r5 = NLPModels.jprod(model, x0, x0)

r = optimize(model)