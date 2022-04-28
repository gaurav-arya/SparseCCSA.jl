#include("../src/SparseCCSA.jl")
include("../src/ccsa.jl")
#include("../src/algo.jl")
#include("../src/model.jl")
using Test
using LinearAlgebra
using CatViews
#using Plots
# min x^2 s.t. x-1<0
function dimensional_1_test()
    function f(x)
        return [sum(abs2,x),x[1]+1.0]
    end
    function ∇f(x) #(m+1)*n
        return [2x[1];1.0;;]
    end
    function f_and_∇f(x)
        return f(x),∇f(x)
    end
    n=1
    m=1
    σ=[30.0] #n
    ρ=[2.0,0.0001]*σ[1]^2 #m+1
    x=[-10.0]
    opt=CCSAState(n,m,f_and_∇f,ρ,σ,x)
    optimize(opt)
    opt.x
end
dimensional_1_test()

##### inner_iterations test ######
function inner_iterations_test()
    function f(x)
        return [sum(abs2,x),x[1]+1.0]
    end
    function ∇f(x) #(m+1)*n
        return [2x[1];1.0;;]
    end
    function f_and_∇f(x)
        return f(x),∇f(x)
    end
    n=1
    m=1
    σ=[30.0] #n
    ρ=[2.0,0.0001]*σ[1]^2 #m+1
    x=[-10.0]
    opt=CCSAState(n,m,f_and_∇f,ρ,σ,x)
    inner_iterations(opt)
    opt.Δx
end
inner_iterations_test()


### test the g_grad of dual_func ###
function inner_iterations_dual_func_test()
    function f(x)
        return [sum(abs2,x),x[1]-1.0]
    end
    function ∇f(x) #(m+1)*n
        return [2x[1];1.0;;]
    end
    function f_and_∇f(x)
        return f(x),∇f(x)
    end
    n=1
    m=1
    σ=[300.0] #n
    ρ=[2.0,0.0000001]*σ[1]^2 #m+1
    x=[-1.0]
    opt=CCSAState(n,m,f_and_∇f,ρ,σ,x)
    #inner_iterations(opt)
    ρ_again=[10.0]
    σ_again=[20.0]
    ###都是-λ-¼*λ²
    max_problem(λ)= begin 
        result=dual_func!(λ,opt) 
        -result[1], -result[2]
    end
    max_problem_template(λ)= begin 
        λ+0.25λ.^2, [1.0 .+ 0.5λ[1];;]
    end
    ###
    p=plot(-100:0.1:100,[max_problem([i])[2][1] for i in -100:0.1:100])
    plot!(p,-100:0.1:100,[max_problem_template([i])[2][1] for i in -100:0.1:100])
    display(p)
end
inner_iterations_dual_func_test()


##### inner_iterations's max_problem test ######
function inner_iterations_max_problem_test()
    function f(x)
        return [sum(abs2,x),x[1]-1.0]
    end
    function ∇f(x) #(m+1)*n
        return [2x[1];1.0;;]
    end
    function f_and_∇f(x)
        return f(x),∇f(x)
    end
    n=1
    m=1
    ρ=[2.0,0.000001]*400 #m+1
    σ=[20.0] #n
    x=[-1.0]
    opt=CCSAState(n,m,f_and_∇f,ρ,σ,x)
    ### test max_problem ###
    max_problem(λ)= begin 
        result=dual_func!(λ,opt) 
        -result[1], -result[2]
    end
    max_problem([10.0])
    # caucilate by hand, max_problem(λ)= -λ-¼*λ²(λ>-2), 1(λ<-2)
    # 不需要看 λ<0 的情况
    # 但是还有信赖域，把它放大，ρ=σ^2
    λ_list=[i for i in -10:0.1:10]
    yy=[max_problem([i])[1][1] for i in -10:0.1:10]
    p=plot(λ_list,yy)
    plot!(p,λ_list,-λ_list-0.25λ_list.^2)
    display(p)
end
inner_iterations_max_problem_test()


############ no constraints problem ##########
function fundamental_no_constraints_test(n)
    function f(x)
        return [sum(abs2,x)]
    end
    function ∇f(x) #(m+1)*n
        return 2*x[:,:]'
    end
    function f_and_∇f(x)
        return f(x),∇f(x)
    end
    m=0
    ρ0=[101.0] #m+1
    σ0=0.1*ones(n) #n
    x0=50.0*ones(n) #n
    st=CCSAState(n,m,f_and_∇f,ρ0,σ0,x0)
    optimize(st)
    return st.x
end
fundamental_no_constraints_test(6)
### max problem ###
function max_no_constraints_test(n)
    function f(x)
        return [-sum(abs2,x)]
    end
    function ∇f(x) #(m+1)*n
        return -2*x[:,:]'
    end
    function f_and_∇f(x)
        return f(x),∇f(x)
    end
    max_problem(λ)= begin 
        result=f_and_∇f(λ) 
        -result[1], -result[2]
    end
    m=0
    ρ0=[10.0] #m+1
    σ0=10.0*ones(n) #n
    x0=50.0*ones(n) #n
    st=CCSAState(n,m,max_problem,ρ0,σ0,x0)
    optimize(st)
    return st.x
end
max_no_constraints_test(6)
### Domain limited problem ###
function max_no_constraints_test(n)
    function f(x)
        return [-sum(abs2,x)]
    end
    function ∇f(x) #(m+1)*n
        return -2*x[:,:]'
    end
    function f_and_∇f(x)
        return f(x),∇f(x)
    end
    m=0
    ρ0=[10.0] #m+1
    σ0=10.0*ones(n) #n
    x0=50.0*ones(n) #n
    st=CCSAState(n,m,f_and_∇f,ρ0,σ0,x0,ones(n))
    optimize(st)
    return st.x
end
max_no_constraints_test(6)