module SparseCCSA

using LinearAlgebra
using UnPack
using StaticArrays

import Base.@kwdef

export init, solve!, step!, reinit!

include("dual.jl")
include("optimize.jl")

end
