# __revise_mode__ = :eval
include("define_rosenbrock.jl")
include("nlopt_df.jl")
include("sparseccsa_df.jl")
using LinearAlgebra

#run_once_nlopt(8)
begin
df_sp = sparseccsa_df(400)
df_n = nlopt_df(334)
df_n_long = nlopt_df(1000)
end
df_sp2 = sparseccsa_df(400)

sparse_n = df_sp.inner_history[7]
inner_n = df_n.inner_history[7]
isapprox(df_n.x[9], df_sp.x[9]; rtol=1e-3)

df_sp.x[30]

# some plotting

using CairoMakie
CairoMakie.activate!()

begin
fig = Figure(resolution = (1200, 600))
# axislegend(ax; position=:rt)
# ρ
# ax = Axis(fig[1, 2]; xlabel="Iterations", yscale=log10)
ax = Axis(fig[1, 1]; xlabel="Iterations", yscale=log10)
lines!(ax, first.(df_sp.ρ); label="SparseCCSA ρ", linestyle=:dash)
lines!(ax, first.(df_n.ρ); label="nlopt ρ", linestyle=:dot)
# σ
lines!(ax, norm.(df_sp.σ); label="SparseCCSA σ", linestyle=:dash)
lines!(ax, norm.(df_n.σ); label="nlopt σ", linestyle=:dot)
axislegend(ax; position=:lt)
#
ax = Axis(fig[1, 2]; xlabel="Iterations")
lines!(ax, first.(df_sp.x); label="SparseCCSA x[1]", linestyle=:dash)
lines!(ax, first.(df_n.x); label="nlopt x[1]", linestyle=:dot)
lines!(ax, last.(df_sp.x); label="SparseCCSA x[2]", linestyle=:dash)
lines!(ax, last.(df_n.x); label="nlopt x[2]", linestyle=:dot)
axislegend(ax; position=:rb)
# lines!(ax, first.(df_n.x); label="nlopt ρ", linestyle=:dot)
# axislegend(ax; position=:rt)
#
ax = Axis(fig[1, 3]; xlabel="Iterations", yscale=log10)
finalx = df_n_long.x[end] .+ 1e-8 # fudge factor for log
lines!(ax, abs.(last.(df_sp.x .- Ref(finalx))); label="SparseCCSA x[2]", linestyle=:dash)
lines!(ax, abs.(last.(df_sp2.x .- Ref(finalx))); label="nlopt x[2]", linestyle=:dot)
lines!(ax, abs.(first.(df_sp.x .- Ref(finalx))); label="SparseCCSA x[1]", linestyle=:dash)
lines!(ax, abs.(first.(df_sp2.x .- Ref(finalx))); label="nlopt x[1]", linestyle=:dot)
axislegend(ax; position=:rt)
fig

end
