module SparseCCSA

using LinearAlgebra
using UnPack
using StaticArrays
using Printf
using DataFrames # TODO: lighter weight alternative?

import Base.@kwdef

export init, solve!, step!, reinit!

include("structs.jl")
include("dual.jl")
include("optimize.jl")

end
