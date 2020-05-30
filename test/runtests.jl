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

track_with_context(model, ctx) = trackdependencies(model, VarInfo(), SampleFromPrior(), ctx)


function testgraph(graph, vns)
    @test varnames(graph) == Set{VarName}(vns)
    @test check_dependencies(graph)
end

macro testmodel(model, varname_exprs...)
    return quote
        let m = $(esc(model)),
            vns = tuple($(DynamicPPL.varname.(varname_exprs)...))
            
            graph_default = trackdependencies(m)
            testgraph(graph_default, vns)
        
            graph_likelihood = track_with_context(m, LikelihoodContext())
            testgraph(graph_likelihood, vns)
        end
    end
end


@testset "AutoGibbs.jl" begin
    @model function test0(x)
        λ ~ Gamma(2.0, inv(3.0))
        m ~ Normal(0, sqrt(1 / λ))
        x ~ Normal(m, sqrt(1 / λ))
    end

    @testmodel(test0(1.4), λ, m, x)


    @model function test1(x)
        s ~ Gamma(1.0, 1.0)
        x .~ Normal(0.0, s)
    end

    
    @testmodel(test1([-0.5, 0.5]), s, x)

    
    @model function test2(x)
        s ~ Gamma(1.0, 1.0)
        for i in eachindex(x)
            x[i] ~ Normal(float(i), s)
        end
    end

    @testmodel(test2([-0.5, 0.5]), s, x[1], x[2])
    
    
    @model function test3(w)
        p ~ Beta(1, 1)
        x ~ Bernoulli(p)
        y ~ Bernoulli(p)
        z ~ Bernoulli(p)
        w ~ MvNormal([x, y, z], 1.2)
    end
    
    @testmodel(test3([1, 0, 1]), p, x, y, z, w)

 
    @model function test4(x) 
        μ ~ MvNormal(fill(0, 2), 2.0)
        z = Vector{Int}(undef, length(x))
        for i in 1:length(x)
            z[i] ~ Categorical([0.5, 0.5])
            x[i] ~ Normal(μ[z[i]], 0.1)
        end
    end
    
    @testmodel(test4([1, 1, -1]), μ, z[1], z[2], z[3], x[1], x[2], x[3])

    
    @model function test5(x) 
        μ ~ MvNormal(fill(0, 2), 2.0)
        z = Vector{Int}(undef, length(x))
        z .~ Categorical.(fill([0.5, 0.5], length(x)))
        for i in 1:length(x)
            x[i] ~ Normal(μ[z[i]], 0.1)
        end
    end

    # there's a bug in DPPL occuring here: https://github.com/TuringLang/DynamicPPL.jl/issues/129
    @testmodel(test5([1, 1, -1]), μ, z, x[1], x[2], x[3])


    @model function test6(x, y)
        λ ~ Gamma(2.0, inv(3.0))
        m ~ Normal(0, sqrt(1 / λ))
        D = Normal(m, sqrt(1 / λ))
        x ~ D
        y ~ D
    end

    @testmodel(test6(1.4, 1.2), λ, m, x, y)

    
    @model function test7(x)
        s ~ Gamma(1.0, 1.0)
        x[1] ~ Normal(0.0, s)
        for i in 2:length(x)
            x[i] ~ Normal(x[i-1], s)
        end
    end

    @testmodel(test7([0.0, 0.1, -0.2]), s, x[1], x[2], x[3])

    
    @model function test8()
        s ~ Gamma(1.0, 1.0)
        0.1 ~ Normal(0.0, s)
    end

    testmodel(test8(), s)

    
    @model function test9(x)
        s ~ Gamma(1.0, 1.0)
        state = zeros(length(x) + 1)
        state[1] ~ Normal(0.0, s)
        for i in 1:length(x)
            state[i+1] ~ Normal(state[i], s)
            x[i] ~ Normal(state[i+1], s)
        end
    end

    @testmodel(test9([0.0, 0.1, -0.2]), s, x[1], x[2], x[3], state[1], state[2], state[3], state[4])


    @model function test10(x)
        s ~ Gamma(1.0, 1.0)
        state = zeros(length(x) + 1)
        state[1] = 42.0
        for i in 1:length(x)
            state[i+1] ~ Normal(state[i], s)
            x[i] ~ Normal(state[i+1], s)
        end
    end

    @testmodel(test10([0.0, 0.1, -0.2]), s, x[1], x[2], x[3], state[2], state[3], state[4])


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

    @testmodel(test11([0.1, 0.05, 1.0]), m, x, z[1], z[2], z[3])
end
