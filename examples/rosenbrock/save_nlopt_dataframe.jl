include("NLoptData.jl")
using .NLoptData
using JLD2

df_n = nlopt_dataframe(1000)
save("examples/rosenbrock/nlopt_dataframe.jld2", Dict("df_n" => df_n))
