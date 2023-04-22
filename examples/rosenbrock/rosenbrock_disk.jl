# __revise_mode__ = :eval
include("define_rosenbrock.jl")
include("solve_sparseccsa.jl")
include("solve_nlopt.jl")

##

try run_once_mine(1) catch e end
special_iters = [5, 7, 9, 11, 13, 14] # iters where nlopt output changes. +1 to see logged outer iter
opt = run_once_mine(4);

run_once_nlopt(nothing, 20)

pipe = Pipe()



opt.iterate.fx[1]
(opt.iterate.x,)
(opt.iterate.ρ,)
(opt.iterate.σ,)

f() = [3,4]

(a,)

##

nlopt.numevals


f([0.5, 0.5])[1]
f([0.603, 0.397])[1]
f(opt.iterate.x)[1]

