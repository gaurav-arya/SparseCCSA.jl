# __revise_mode__ = :eval
include("define_rosenbrock.jl")
include("solve_nlopt.jl")
include("solve_sparseccsa.jl")

df_n = nlopt_df(1200)
df_sp = sparseccsa_df(500)

df_n[3, :]
df_sp[3, :]
df_n.inner_history[4]
df_sp.inner_history[4]
df_n[4, :]
df_sp[4, :]

using GLMakie
lines(first.(df_n.σ))
lines(first.(df_sp.σ))
lines(log.(first.(df_n.ρ)))
lines(log.(first.(df_sp.ρ)))
lines(log.(last.(df_sp.ρ)))

