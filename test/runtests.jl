using Test

using AutoGibbs
using Distributions
using DynamicPPL
using Turing, Turing.RandomMeasures


function varnames(graph)
    vars = (ref for ref in keys(graph) if ref isa AutoGibbs.NamedReference)
    return Set(AutoGibbs.resolve_varname(graph, ref) for ref in vars)
end

check_dependencies(graph) = all(haskey(graph, dep)
                                for expr in values(graph)
                                for dep in AutoGibbs.dependencies(expr))


@testset "AutoGibbs.jl" begin
    @model function test0(x)
        λ ~ Gamma(2.0, inv(3.0))
        m ~ Normal(0, sqrt(1 / λ))
        x ~ Normal(m, sqrt(1 / λ))
    end

    graph0 = trackdependencies(test0(1.4))
    @test varnames(graph0) == Set([@varname(λ), @varname(m), @varname(x)])
    @test check_dependencies(graph0)


    @model function test1(x)
        s ~ Gamma(1.0, 1.0)
        x .~ Normal(0.0, s)
    end

    graph1 = trackdependencies(test1([-0.5, 0.5]))
    @test varnames(graph1) == Set([@varname(s), @varname(x)])
    @test check_dependencies(graph1)

    
    @model function test2(x)
        s ~ Gamma(1.0, 1.0)
        for i in eachindex(x)
            x[i] ~ Normal(float(i), s)
        end
    end

    graph2 = trackdependencies(test2([-0.5, 0.5]))
    @test varnames(graph2) == Set([@varname(s), @varname(x[1]), @varname(x[2])])
    @test check_dependencies(graph2)
    
    
    @model function test3(w)
        p ~ Beta(1, 1)
        x ~ Bernoulli(p)
        y ~ Bernoulli(p)
        z ~ Bernoulli(p)
        w ~ MvNormal([x, y, z], 1.2)
    end
    
    graph3 = trackdependencies(test3([1, 0, 1]))
    @test varnames(graph3) == Set([@varname(p), @varname(x), @varname(y), @varname(y),
                                   @varname(z), @varname(w)])
    @test check_dependencies(graph3)

    
    @model function test4(x) 
        μ ~ MvNormal(fill(0, 2), 2.0)
        z = Vector{Int}(undef, length(x))
        for i in 1:length(x)
            z[i] ~ Categorical([0.5, 0.5])
            x[i] ~ Normal(μ[z[i]], 0.1)
        end
    end
    
    graph4 = trackdependencies(test4([1, 1, -1]))
    @test varnames(graph4) == Set([@varname(μ),
                                   @varname(z[1]), @varname(z[2]), @varname(z[3]),
                                   @varname(x[1]), @varname(x[2]), @varname(x[3])])
    @test check_dependencies(graph4)

    
    @model function test5(x) 
        μ ~ MvNormal(fill(0, 2), 2.0)
        z = Vector{Int}(undef, length(x))
        z .~ Categorical.(fill([0.5, 0.5], length(x)))
        for i in 1:length(x)
            x[i] ~ Normal(μ[z[i]], 0.1)
        end
    end
    
    graph5 = trackdependencies(test5([1, 1, -1]))
    @test varnames(graph5) == Set([@varname(μ), @varname(z),
                                   @varname(x[1]), @varname(x[2]), @varname(x[3])])
    @test check_dependencies(graph5)


    @model function test6(x, y)
        λ ~ Gamma(2.0, inv(3.0))
        m ~ Normal(0, sqrt(1 / λ))
        D = Normal(m, sqrt(1 / λ))
        x ~ D
        y ~ D
    end

    graph6 = trackdependencies(test6(1.4, 1.2))
    @test varnames(graph6) == Set([@varname(λ), @varname(m), @varname(x), @varname(y)])
    @test check_dependencies(graph6)

    
    @model function test7(x)
        s ~ Gamma(1.0, 1.0)
        x[1] ~ Normal(0.0, s)
        for i in 2:length(x)
            x[i] ~ Normal(x[i-1], s)
        end
    end

    graph7 = trackdependencies(test7([0.0, 0.1, -0.2]))
    @test varnames(graph7) == Set([@varname(s), @varname(x[1]), @varname(x[2]), @varname(x[3])])
    @test check_dependencies(graph7)

    
    @model function test8()
        s ~ Gamma(1.0, 1.0)
        0.1 ~ Normal(0.0, s)
    end

    graph8 = trackdependencies(test8())
    @test varnames(graph8) == Set([@varname(s)])
    @test check_dependencies(graph8)

    
    @model function test9(x)
        s ~ Gamma(1.0, 1.0)
        state = zeros(length(x) + 1)
        state[1] ~ Normal(0.0, s)
        for i in 1:length(x)
            state[i+1] ~ Normal(state[i], s)
            x[i] ~ Normal(state[i+1], s)
        end
    end

    graph9 = trackdependencies(test9([0.0, 0.1, -0.2]))
    @test varnames(graph9) == Set([@varname(s),
                                   @varname(x[1]), @varname(x[2]), @varname(x[3]),
                                   @varname(state[1]), @varname(state[2]), @varname(state[3]),
                                   @varname(state[4])])
    @test check_dependencies(graph9)


    @model function test10(x)
        s ~ Gamma(1.0, 1.0)
        state = zeros(length(x) + 1)
        state[1] = 42.0
        for i in 1:length(x)
            state[i+1] ~ Normal(state[i], s)
            x[i] ~ Normal(state[i+1], s)
        end
    end

    graph10 = trackdependencies(test10([0.0, 0.1, -0.2]))
    @test varnames(graph10) == Set([@varname(s),
                                   @varname(x[1]), @varname(x[2]), @varname(x[3]),
                                   @varname(state[2]), @varname(state[3]), @varname(state[4])])
    @test check_dependencies(graph10)


    @model function test11(x)
        rpm = DirichletProcess(1.0)
        n = zeros(Int, length(x))
        z = zeros(Int, length(x))
        for i in eachindex(x)
            z[i] ~ ChineseRestaurantProcess(rpm, n)
            n[z[i]] += 1
        end
        K = findlast(!iszero, n)
        m ~ MvNormal(fill(0.0, K), 1.0)
        x ~ MvNormal(m[z], 1.0)
    end

    graph11 = trackdependencies(test11([0.1, 0.05, 1.0]))
    @test varnames(graph11) == Set([@varname(m), @varname(x),
                                    @varname(z[1]), @varname(z[2]), @varname(z[3])])
    @test check_dependencies(graph11)

end
