include("../src/RosenbrockExample.jl")
include("nlopt_dataframe.jl")
using .RosenbrockExample
using LinearAlgebra

begin
df_sp = sparseccsa_dataframe(400)
df_n = nlopt_dataframe(334)
df_n_long = nlopt_dataframe(1000)
end

# Plot NLopt vs SparseCCSA

using CairoMakie

begin
fig = Figure(resolution = (1200, 600))
# axislegend(ax; position=:rt)
# ρ
# ax = Axis(fig[1, 2]; xlabel="Iterations", yscale=log10)
ax = Axis(fig[1, 1]; xlabel="Iterations", ylabel="log(value)", yscale=log10)
lines!(ax, first.(df_sp.ρ); label="SparseCCSA ρ", linestyle=:dash)
lines!(ax, first.(df_n.ρ); label="nlopt ρ", linestyle=:dot)
# σ
lines!(ax, norm.(df_sp.σ); label="SparseCCSA σ", linestyle=:dash)
lines!(ax, norm.(df_n.σ); label="nlopt σ", linestyle=:dot)
axislegend(ax; position=:lt)
#
ax = Axis(fig[1, 2]; xlabel="Iterations", ylabel="value")
lines!(ax, first.(df_sp.x); label="SparseCCSA x[1]", linestyle=:dash)
lines!(ax, first.(df_n.x); label="nlopt x[1]", linestyle=:dot)
lines!(ax, last.(df_sp.x); label="SparseCCSA x[2]", linestyle=:dash)
lines!(ax, last.(df_n.x); label="nlopt x[2]", linestyle=:dot)
axislegend(ax; position=:rb)
# lines!(ax, first.(df_n.x); label="nlopt ρ", linestyle=:dot)
# axislegend(ax; position=:rt)
#
ax = Axis(fig[1, 3]; xlabel="Iterations", ylabel="log(error)", yscale=log10)
finalx = df_n_long.x[end] .+ 1e-8 # fudge factor for log
lines!(ax, abs.(last.(df_sp.x .- Ref(finalx))); label="SparseCCSA x[2]", linestyle=:dash)
lines!(ax, abs.(last.(df_n.x .- Ref(finalx))); label="nlopt x[2]", linestyle=:dot)
lines!(ax, abs.(first.(df_sp.x .- Ref(finalx))); label="SparseCCSA x[1]", linestyle=:dash)
lines!(ax, abs.(first.(df_n.x .- Ref(finalx))); label="nlopt x[1]", linestyle=:dot)
axislegend(ax; position=:rt)
fig

end
