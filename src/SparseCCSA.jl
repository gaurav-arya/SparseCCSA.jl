module SparseCCSA

using CatViews
using LinearAlgebra
using UnPack

import Base.@kwdef

export init, solve!

include("ccsa.jl")

end
