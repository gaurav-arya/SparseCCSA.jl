module SparseCCSA

using LazyArrays
using LinearAlgebra
using UnPack
using StaticArrays
using CatViews

import Base.@kwdef

export init, solve!, step!, reinit!

include("dual.jl")
include("optimize.jl")

end
