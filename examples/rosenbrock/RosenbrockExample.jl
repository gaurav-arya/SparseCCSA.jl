module RosenbrockExample

using LinearAlgebra
using ForwardDiff
using SparseCCSA
using NLopt

include("define_rosenbrock.jl")
include("nlopt_dataframe.jl")
include("sparseccsa_dataframe.jl")

export nlopt_dataframe, sparseccsa_dataframe

end