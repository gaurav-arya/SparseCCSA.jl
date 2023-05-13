# https://www.mathworks.com/help/optim/ug/example-nonlinear-constrained-minimization.html?w.mathworks.com
module DefineRosenbrock

using ForwardDiff

function f(x)
    # obj = x[1]^4#
    obj = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
    # cons = x[1]^2 - 1
    cons = x[1]^2 + x[2]^2 - 1
    return [obj, cons]
end

function f_and_jac(fx, jac, x)
    fx .= f(x)
    if jac !== nothing
        jac .= ForwardDiff.jacobian(f, x)
    end
    nothing
end

n = 2
m = 1

# example usage
# fx = zeros(m + 1)
# jac = zeros(m + 1, n)
# f_and_jac(fx, jac, zeros(n))

export f, f_and_jac, n, m

end