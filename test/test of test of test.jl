using SparseCCSA
# Problem:
# minimize x₁²+x₂²+x₃²
# subject to x₁ <= -1
# x∈Domain=R×R×[5,15]
n=3 # number of variables
m=1 # number of constraints
function f_and_∇f(x)
    f=[sum(abs2,x),x[1]+1]
    ∇f=[2x[1] 2x[2] 2x[3]; 1.0 0.0 0.0]
    return f,∇f #f is a length m+1 vector, ∇f is a (m+1)*n matrix
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
