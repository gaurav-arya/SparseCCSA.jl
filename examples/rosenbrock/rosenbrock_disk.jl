# __revise_mode__ = :eval
include("define_rosenbrock.jl")
include("solve_nlopt.jl")
include("solve_sparseccsa.jl")

run_once_nlopt(8)
df_sp = sparseccsa_df(2)

df_sp.inner_history[2]

df_n = nlopt_df(10)
df_sp = sparseccsa_df(2)

df_n
df_sp[1, :]
sparse_n = df_sp.inner_history[2]
inner_n = df_n.inner_history[2]
df_n.inner_history[2].x_proposed[end] - df_n.inner_history[1].x_proposed[end]

fx = zeros(2)
jac_fx = zeros(2, 2)
f_and_jac(fx, jac_fx, df_sp.x[1])
jac_fx

inner3_n.ρ .- sparse3_n.ρ





# some plotting

using GLMakie
lines(first.(df_n.σ))
lines(first.(df_sp.σ))
lines(log.(first.(df_n.ρ)))
lines(log.(first.(df_sp.ρ)))
lines(log.(last.(df_sp.ρ)))

