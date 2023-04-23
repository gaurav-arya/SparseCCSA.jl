# __revise_mode__ = :eval
include("define_rosenbrock.jl")
include("solve_nlopt.jl")
include("solve_sparseccsa.jl")

df_n = nlopt_df(30)

run_once_nlopt(7)
Base.Libc.flush_cstdio()

df_n[1:5, :]
df_sp = sparseccsa_df(3)

df_sp[1,:].inner_history
df_sp[3,:].x - df_sp[2,:].x 
df_sp[2,:].inner_history
x2_n - df_sp[1,:].x ≈ df_sp[2,:].inner_history.Δx_proposed[end]

##
df_sp[3,:].inner_history.Δx_proposed[end]
df_sp[3,:].x - df_sp[2,:].x ≈ df_sp[3,:].inner_history.Δx_proposed[end]
x3_n - df_sp[2,:].x
##
df_sp[3,:].inner_history
df_n[3,:].inner_history
x3_n


df_sp[1, :].x
x1_n
x2_n ≈ df_sp[2,:].x
x3_n - df_sp[3,:].x
df_n[2,:].inner_history
df_n[3, :]

using GLMakie
lines(first.(df_n.σ))
lines(first.(df_sp.σ))
lines(log.(first.(df_n.ρ)))
lines(log.(first.(df_sp.ρ)))
lines(log.(last.(df_sp.ρ)))

