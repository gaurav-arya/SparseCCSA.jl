include("define_rosenbrock.jl")
include("solve_sparseccsa.jl")

df_sp = sparseccsa_df(100) # a DataFrame containing optimization history for the first 100 iterations
