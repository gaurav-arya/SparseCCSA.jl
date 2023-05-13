include("../src/RosenbrockExample.jl")
using .RosenbrockExample
using CSV

df_n = nlopt_dataframe(1000)
CSV.write("examples/rosenbrock/df_n.csv", df_n)