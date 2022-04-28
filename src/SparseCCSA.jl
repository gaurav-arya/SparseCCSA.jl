module SparseCCSA
    using CatViews
    using LinearAlgebra
    export CCSAState
    export dual_func! 
    export optimize
    include("ccsa.jl")
end
