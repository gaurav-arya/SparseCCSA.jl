# copied from https://github.com/gaurav-arya/ImplicitAdjoints.jl/
module FISTASolver
__revise_mode__ = :eval

using LinearAlgebra
using IterativeSolvers
using LinearMaps

# subtypes should implement:
# transform_op(x, reg) -> LinearOperator
# projection_op(supp, reg) -> LinearOperator
# proximal!(x, thresh, reg) -> vector
# lipschitz_scale(reg) -> value
# support(x, reg) -> vector
# TODO: document above properly
abstract type Regularizer end

# uses the Fast Iterative Soft-Thresholding Algorithm (FISTA) to
# minimize f(x) + g(x) = ½(|Ax - y|² + β|x|²) + ½α|Ψx|₁

fista(A, y, α, β, iters, tol, reg::Regularizer) = fista(A, y, α, β, iters, tol, reg::Regularizer, zeros(size(A)[2]))

function fista(A, y, α, β, iters, tol, reg::Regularizer, xstart)
    n, p = size(A)
    
    maxeig = abs(powm(A' * A, maxiter=100)[1]) # TODO: handle number of eigiters
    L = maxeig + β # Lipschitz constant of f (strongly convex term)
    η = 1 / (L * lipschitz_scale(reg))

    x = xstart[:]
    z = x[:] 
    xold = similar(x)
    res = similar(y) 
    grad = similar(x)
    t = 1
    
    iters_done = iters
    xupdates = Float64[] 
    convdists = Float64[]
    evals = Float64[]

    Ψ = transform_op(reg)
    Fold = Inf

    for i = 1:iters

        xold .= x

        res .= mul!(res, A, z) .- y # TODO: maybe use five-arg mul! instead. (but IterativeSolvers sticks to three-arg)
        grad .= mul!(grad, A', res) .+ β .* z 

        x .= z .- η .* grad
        proximal!(x, 1/2 * η * α, reg)

        restart = dot(z .- x, x .- xold)
        restart > 0 && (t = 1)

        told = t
        t = 1/2 * (1 + √(1 + 4t^2))
        
        z .= x .+ (told - 1)/t .* (x .- xold)

        xupdate = norm(z .- xold) / norm(xold)
        append!(xupdates, xupdate)

        if xupdate < tol
            iters_done = i
            break
        end
    end

    x, (;iters=iters_done, final_tol=norm(x .- xold) / norm(x), xupdates) 
end

## regularizer implementations

support(x, thresh, reg::Regularizer) = abs.(transform_op(reg) * x) .> thresh
proximal(x, thresh, reg::Regularizer) = proximal!(copy(x), thresh, reg) # fallback for out-of-place proximal

## L1 regularizer
struct L1 <: Regularizer 
    size::Int
end

support(x, reg::L1) = support(x, 1e-3, reg)
lipschitz_scale(::L1) = 2.

# proximal

softthresh(val, thresh) = max(abs(val) - thresh, 0) .* sign(val)
function proximal!(x, thresh, reg::L1)
    x .= softthresh.(x, thresh)
    x
end

# transform

transform_op(reg::L1) = LinearMap(x -> x, x -> x, reg.size, reg.size)

# projection

struct L1Project <: LinearMap{Float64}
    p::Int
    suppinv::AbstractVector{<:Int}
end

Base.size(P::L1Project) = (sum(P.suppinv .> 0), P.p)
L1ProjectTranspose = LinearMaps.TransposeMap{<:Any, <:L1Project}

function L1Project(p::Int, supp::AbstractVector{Bool})
    suppinv = zeros(Int, p)
    index = 1
    for i in 1:p
        if supp[i]
            suppinv[i] = index
            index += 1
        end
    end
    L1Project(p, suppinv)
end

Base.:(*)(P::L1Project, x::AbstractVector) = x[P.suppinv .> 0]
Base.:(*)(Pt::L1ProjectTranspose, y::AbstractVector) = [Pt.lmap.suppinv[i] > 0 ? y[Pt.lmap.suppinv[i]] : 0.0 for i in 1:Pt.lmap.p] # TODO: use zero element instead?
LinearAlgebra.mul!(uflat::AbstractVecOrMat, P::L1Project, yflat::AbstractVector) = (uflat[:] = P * yflat)
LinearAlgebra.mul!(uflat::AbstractVecOrMat, Pt::L1ProjectTranspose, yflat::AbstractVector) = (uflat[:] = Pt * yflat)

projection_op(supp, reg::L1) = L1Project(reg.size, supp)

## TV regularizer

# TODO: type-genericness?
struct TVProximalWork
    Δ::AbstractArray{Float64}
    diff::AbstractArray{Float64}
end

struct TV <: Regularizer 
    size::Tuple
    wrap::Bool
    work::TVProximalWork
end

TV(size::Tuple) = TV(size, true)
function TV(size::Tuple, wrap::Bool)
    Δ = Array{Float64}(undef, size)
    diff = Array{Float64}(undef, size)
    TV(size, wrap, TVProximalWork(Δ, diff))
end

support(x, reg::TV) = support(x, 0.005, reg)
lipschitz_scale(::TV) = 50

# proximal

function proximal!(x, thresh, reg::TV)
    x = reshape(x, reg.size)
    D = length(reg.size)
    Δ = reg.work.Δ
    diff = reg.work.diff
    Δ .= 0
    for i in 1:D
        xroll = roll(x, -1, i)
        diff .= (d -> min(2 * D * thresh, abs(d) / 2) * sign(d)).(xroll .- x)
        Δ .+= diff ./ (2 * D)
        Δ .-= roll(diff, 1, i) ./  (2 * D)
    end
    x .+= Δ
    reshape(x, prod(reg.size))
end

function proximal2(x, thresh, reg::TV)
    y = x[:]
    proxTV!(y, thresh, shape=reg.size, iterations=20)
    y
end

# transform

roll(x, shift, dim) = ShiftedArrays.circshift(x, (zeros(dim-1)..., shift, zeros(ndims(x)-dim)...))

struct Gradient <: LinearMap{Float64}
    size::Tuple
    dim::Int
    wrap::Bool
end

Base.size(D::Gradient) = (prod(D.size), prod(D.size))
GradientTranspose = LinearMaps.TransposeMap{<:Any, <:Gradient}
function Base.:(*)(D::Gradient, xflat::AbstractVector)
    x = reshape(xflat, D.size)
    y = x .- roll(x, -1, D.dim)
    !D.wrap && (y[end] = 0)
    y[:]
end
function Base.:(*)(Dt::GradientTranspose, xflat::AbstractVector)
    D = Dt.lmap
    x = reshape(xflat, D.size)
    y = x .- roll(x, 1, D.dim)
    !D.wrap && (y[1] = 0)
    y[:]
end

LinearAlgebra.mul!(uflat::AbstractVecOrMat, D::Gradient, yflat::AbstractVector) = (uflat[:] = D * yflat)
LinearAlgebra.mul!(uflat::AbstractVecOrMat, Dt::GradientTranspose, yflat::AbstractVector) = (uflat[:] = Dt * yflat)

function transform_op(reg::TV)
    ndims = length(reg.size)
    vcat([Gradient(reg.size, dim, reg.wrap) for dim in 1:ndims]...) # TODO: (style) should I avoid these sort of splatted arrays?
end

# projection

# TODO (relevant everywhere): proper typing for things like 2D arrays
# TODO: type generic code (no Float64's)
struct TVProject{N} <: LinearMap{Float64}
    size::NTuple{N, Int}
    regions::AbstractVector{AbstractVector{CartesianIndex{N}}}
end

Base.size(P::TVProject) = (length(P.regions), prod(P.size))
TVProjectTranspose = LinearMaps.TransposeMap{<:Any, <:TVProject}

function TVProject(size::Tuple, regions::AbstractVector{AbstractVector{CartesianIndex}})
    TVProject{length(size)}(size, regions) # TODO: check if this what I'm supposed to do
end

# TODO: type support properly
function TVProject(size::Tuple, supp::AbstractVector{<:Bool}, wrap::Bool)
    ndims = length(size)
    explored = fill(false, size)
    regions = AbstractVector{CartesianIndex{ndims}}[]
    supp = reshape(supp, (size..., ndims))
    for start_pt in CartesianIndices(size)
        explored[start_pt] && continue
        region = [start_pt]
        explored[start_pt] = true;
        s = Stack{CartesianIndex{ndims}}();
        push!(s, start_pt)
        while !isempty(s)
            pt = pop!(s)
            pttup = Tuple(pt)
            for i in 1:ndims
                if wrap || pt[i] < size[i]
                    pt_right = CartesianIndex(pttup[1:i-1]..., 1 + mod(pt[i], size[i]), pttup[i+1:ndims]...)
                    if !supp[pt, i] && !explored[pt_right]
                        explored[pt_right] = true
                        push!(s, pt_right)
                        push!(region, pt_right)
                    end
                end
                if wrap || pt[i] > 1
                    pt_left = CartesianIndex(pttup[1:i-1]..., 1 + mod(pt[i]-2, size[i]), pttup[i+1:ndims]...)
                    if !supp[pt_left, i] && !explored[pt_left]
                        explored[pt_left] = true
                        push!(s, pt_left)
                        push!(region, pt_left)
                    end
                end
            end
        end
        push!(regions, region)
    end
    TVProject(size, regions)
end

function Base.:(*)(P::TVProject, x::AbstractVector)
    x = reshape(x, P.size)
    region_norm(region) = sum(x[pt] for pt in region) / sqrt(length(region))
    map(region_norm, P.regions)
end

function region_vals(P::TVProject, x::AbstractVector)
    x = reshape(x, P.size)
    function region_vals(region) 
        vals = [x[pt] for pt in region]
        sum(vals) / length(region), maximum(vals) - minimum(vals), length(vals)
    end
    map(region_vals, P.regions)
end

function Base.:(*)(Pt::TVProjectTranspose, y::AbstractVector) 
    P = Pt.lmap
    x = zeros(P.size)
    for (val, region) in zip(y, P.regions)
        x[region] .= val / sqrt(length(region))
    end
    x[:]
end

projection_op(supp, reg::TV) = TVProject(reg.size, supp, reg.wrap)
LinearAlgebra.mul!(uflat::AbstractVecOrMat, P::TVProject, yflat::AbstractVector) = (uflat[:] = P * yflat)
LinearAlgebra.mul!(uflat::AbstractVecOrMat, Pt::TVProjectTranspose, yflat::AbstractVector) = (uflat[:] = Pt * yflat)

export fista, L1, TV

end