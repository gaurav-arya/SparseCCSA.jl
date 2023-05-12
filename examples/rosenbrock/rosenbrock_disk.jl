# __revise_mode__ = :eval
include("define_rosenbrock.jl")
include("solve_nlopt.jl")
include("solve_sparseccsa.jl")

#run_once_nlopt(8)
df_sp = sparseccsa_df(100)
df_n = nlopt_df(230)

sparse_n = df_sp.inner_history[7]
inner_n = df_n.inner_history[7]
isapprox(df_n.inner_history[9].x_proposed[end], df_sp.inner_history[9].x_proposed[end]; rtol=1e-3)

# some plotting

using CairoMakie
CairoMakie.activate!()

begin
fig = Figure()
# σ
ax = Axis(fig[1, 1]; xlabel="Iterations", yscale=log10)
lines!(ax, first.(df_sp.σ); label="SparseCCSA σ", linestyle=:dash)
lines!(ax, first.(df_n.σ); label="nlopt σ", linestyle=:dot)
axislegend(ax; position=:rt)
# ρ
ax = Axis(fig[1, 2]; xlabel="Iterations", yscale=log10)
lines!(ax, first.(df_sp.ρ); label="SparseCCSA ρ", linestyle=:dash)
lines!(ax, first.(df_n.ρ); label="nlopt ρ", linestyle=:dot)
axislegend(ax; position=:rt)
#
fig
end
# makie add legend to ax

fig

lines(log.(first.(df_n.ρ)))
lines(log.(first.(df_sp.ρ)))
lines(log.(last.(df_sp.ρ)))

