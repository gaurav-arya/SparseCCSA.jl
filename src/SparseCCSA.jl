module SparseCCSA

using LazyArrays
using LinearAlgebra
using UnPack
using StaticArrays

import Base.@kwdef

export init, solve!, step!

include("dual.jl")
include("optimize.jl")

end
