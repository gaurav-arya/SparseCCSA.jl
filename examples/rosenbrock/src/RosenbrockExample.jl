module RosenbrockExample

using ForwardDiff
using SparseCCSA
using NLopt

include("define_rosenbrock.jl")
include("sparseccsa_dataframe.jl")

export nlopt_dataframe, sparseccsa_dataframe

end