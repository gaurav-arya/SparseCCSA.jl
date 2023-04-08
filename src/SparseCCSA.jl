module SparseCCSA

using LinearAlgebra
using UnPack
using StaticArrays
using Printf

import Base.@kwdef

export init, solve!, step!, reinit!

include("dual.jl")
include("optimize.jl")

end
