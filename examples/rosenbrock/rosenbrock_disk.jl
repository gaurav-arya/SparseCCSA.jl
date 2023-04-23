# __revise_mode__ = :eval
include("define_rosenbrock.jl")
include("solve_nlopt.jl")
include("solve_sparseccsa.jl")

nlopt_df(20).d_inner[1]
sparseccsa_df(5).inner_history[1]
