# SparseCCSA
## Tutorial

The following example code solves the nonlinearly constrained minimization problem
<img src="https://latex.codecogs.com/svg.latex?\min_{x\in\mathbb{R}\times\mathbb{R}\times[5,15]}\quad%20x_1^2+x_2^2+x_3^2\quad\textrm{s.t.%20}x_1\leq-1">

```julia
using SparseCCSA
n=3 # number of variables
m=1 # number of constraints
function f_and_∇f(x)
    f=[sum(abs2,x),x[1]+1]
    ∇f=[2x[1] 2x[2] 2x[3]; 1.0 0.0 0.0]
    return f,∇f
end
function cb() #callback
    println(opt.x)
end
x=[-1000.0,-1000.0,10.0]
lb=[-Inf,-Inf,5.0]
ub=[Inf,Inf,15.0]
opt=CCSAState(n,m,f_and_∇f,x,lb=lb,ub=ub,max_iters=1000)
optimize(opt,callback=cb)
println("got $(opt.fx[1]) at $(opt.x) after $(opt.iters) iterations")
```
The output should be:
```
got 26.00001803422383 at [-0.9999999999999751, 2.5679858148287025e-7, 5.0] after 13 iterations
```
