using SparseCCSA

# Simple problem:
# maximize x1^2 * x_2
# subject to x_1 <= 3
# and x_1 <= -2
# and x_2 <= 4

function f_and_grad(x)
    fx = [x[1]^2 ^ x[2], x[1] - 3, x[1] + 4, x[2] - 4]
    gradx = [2 * x[1] 1; 1 0; 1 0; 0 1]
    fx, gradx
end

st = CCSAState(
          2,
          3,
          [0,0],
          [100,100],
          f_and_grad,
          # the stuff below shouldn't matter
          # later we will ensure that the user does not have to create these, but don't worry about it for now.
          ones(m+1) 
          ones(n),
          zeros(n),
          zeros(m+1),
          zeros(m+1,n),

          zeros(n),
          zeros(n),
          zeros(n),
          zeros(n),
          zeros(n),
          zeros(m+1)
         )

optimize(st) # get answer
