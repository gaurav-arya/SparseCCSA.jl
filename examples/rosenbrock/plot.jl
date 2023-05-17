includet("SparseCCSARosenbrockData.jl")
includet("NLoptRosenbrockData.jl")
using .SparseCCSARosenbrockData
using .NLoptRosenbrockData
using LinearAlgebra

begin
df_sp = sparseccsa_dataframe(60)
df_n = nlopt_dataframe(210)
df_n_long = nlopt_dataframe(1000)
end

# Plot NLopt vs SparseCCSA

using CairoMakie

begin
fig = Figure(resolution = (1200, 600))
# axislegend(ax; position=:rt)
# ρ
# ax = Axis(fig[1, 2]; xlabel="Iterations", yscale=log10)
ax = Axis(fig[1, 1]; xlabel="Iterations", ylabel="log(value)", yscale=log10, title="a)")
ylims!(ax, 1e-2, 1e5)
lines!(ax, norm.(df_sp.ρ); label="SparseCCSA norm(ρ)", linestyle=:dash)
lines!(ax, norm.(df_n.ρ); label="nlopt norm(ρ)", linestyle=:dot)
# σ
lines!(ax, norm.(df_sp.σ); label="SparseCCSA norm(σ)", linestyle=:dash)
lines!(ax, norm.(df_n.σ); label="nlopt norm(σ)", linestyle=:dot)
axislegend(ax; position=:lb)
#
ax = Axis(fig[1, 2]; xlabel="Iterations", ylabel="value", title="b)")
lines!(ax, first.(df_sp.x); label="SparseCCSA x[1]", linestyle=:dash)
lines!(ax, first.(df_n.x); label="nlopt x[1]", linestyle=:dot)
lines!(ax, last.(df_sp.x); label="SparseCCSA x[2]", linestyle=:dash)
lines!(ax, last.(df_n.x); label="nlopt x[2]", linestyle=:dot)
axislegend(ax; position=:rb)
# lines!(ax, first.(df_n.x); label="nlopt ρ", linestyle=:dot)
# axislegend(ax; position=:rt)
#
ax = Axis(fig[1, 3]; xlabel="Iterations", ylabel="log(error)", yscale=log10, title="c)")

# finalx = df_n_long.x[end] .+ 1e-8 # fudge factor for log
finalx = [0.7864151541684278300665459428, 0.6176983125233934845675555]

lines!(ax, abs.(last.(df_sp.x .- Ref(finalx))); label="SparseCCSA x[2]", linestyle=:dash)
lines!(ax, abs.(last.(df_n.x .- Ref(finalx))); label="nlopt x[2]", linestyle=:dot)
lines!(ax, abs.(first.(df_sp.x .- Ref(finalx))); label="SparseCCSA x[1]", linestyle=:dash)
lines!(ax, abs.(first.(df_n.x .- Ref(finalx))); label="nlopt x[1]", linestyle=:dot)
axislegend(ax; position=:lb)

save("../plots/rosenbrock_compare.png", fig; px_per_unit=5)
fig
end
