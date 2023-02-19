module SparseCCSA

using CatViews
using LinearAlgebra
using UnPack

import Base.@kwdef

export init, solve!, step!

include("dual.jl")
include("optimize.jl")

end
