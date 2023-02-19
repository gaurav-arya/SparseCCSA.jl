function f()
    local x
    for i in 1:3
        x = 1
    end
    return x
end

using InteractiveUtils
@code_warntype f()
println(f())
