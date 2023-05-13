module RosenbrockExample

using ForwardDiff
using SparseCCSA

include("define_rosenbrock.jl")
include("sparseccsa_dataframe.jl")

export sparseccsa_dataframe

end