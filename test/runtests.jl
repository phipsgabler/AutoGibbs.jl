using AutoGibbs
using Test

using Distributions
using DynamicPPL


varnames(graph) = Set(tilde.vn
                      for tilde in values(graph)
                      if tilde isa Union{AutoGibbs.Assumption, AutoGibbs.Observation})

@testset "AutoGibbs.jl" begin
    @model function test0(x)
        λ ~ Gamma(2.0, inv(3.0))
        m ~ Normal(0, sqrt(1 / λ))
        x ~ Normal(m, sqrt(1 / λ))
    end

    t0 = trackmodel(test0(1.4)) |> AutoGibbs.strip_calls
    d0 = dependencies(t0)
    @test varnames(makegraph(d0)) == Set([@varname(λ), @varname(m), @varname(x)])


    @model function test1(x)
        s ~ Gamma(1.0, 1.0)
        x .~ Normal(0.0, s)
    end

    t1 = trackmodel(test1([-0.5, 0.5])) |> AutoGibbs.strip_calls
    d1 = dependencies(t1)
    @test varnames(makegraph(d1)) == Set([@varname(s), @varname(x)])
    
    
    @model function test2(x)
        s ~ Gamma(1.0, 1.0)
        for i in eachindex(x)
            x[i] ~ Normal(float(i), s)
        end
    end

    t2 = trackmodel(test2([-0.5, 0.5])) |> AutoGibbs.strip_calls
    d2 = dependencies(t2)
    @test varnames(makegraph(d2)) == Set([@varname(s), @varname(x[1]), @varname(x[2])])
    
    
    @model function test3(w)
        p ~ Beta(1, 1)
        x ~ Bernoulli(p)
        y ~ Bernoulli(p)
        z ~ Bernoulli(p)
        w ~ MvNormal([x, y, z], 1.2)
    end
    
    t3 = trackmodel(test3([1, 0, 1])) |> AutoGibbs.strip_calls
    d3 = dependencies(t3)
    @test varnames(makegraph(d3)) == Set([@varname(p), @varname(x), @varname(y), @varname(y),
                                          @varname(z), @varname(w)])
    
    
    @model function test4(x) 
        μ ~ MvNormal(fill(0, 2), 2.0)
        z = Vector{Int}(undef, length(x))
        for i in 1:length(x)
            z[i] ~ Categorical([0.5, 0.5])
            x[i] ~ Normal(μ[z[i]], 0.1)
        end
    end
    
    t4 = trackmodel(test4([1, 1, -1])) |> AutoGibbs.strip_calls
    d4 = dependencies(t4)
    @test varnames(makegraph(d4)) == Set([@varname(μ),
                                          @varname(z[1]), @varname(z[2]), @varname(z[3]),
                                          @varname(x[1]), @varname(x[2]), @varname(x[3])])
    
    
    @model function test5(x) 
        μ ~ MvNormal(fill(0, 2), 2.0)
        z = Vector{Int}(undef, length(x))
        z .~ Categorical.(fill([0.5, 0.5], length(x)))
        for i in 1:length(x)
            x[i] ~ Normal(μ[z[i]], 0.1)
        end
    end
    
    t5 = trackmodel(test5([1, 1, -1])) |> AutoGibbs.strip_calls
    d5 = dependencies(t5)
    @test varnames(makegraph(d5)) == Set([@varname(μ), @varname(z),
                                          @varname(x[1]), @varname(x[2]), @varname(x[3])])


    @model function test6(x, y)
        λ ~ Gamma(2.0, inv(3.0))
        m ~ Normal(0, sqrt(1 / λ))
        D = Normal(m, sqrt(1 / λ))
        x ~ D
        y ~ D
    end

    t6 = trackmodel(test6(1.4, 1.2)) |> AutoGibbs.strip_calls
    d6 = dependencies(t6)
    @test varnames(makegraph(d6)) == Set([@varname(λ), @varname(m), @varname(x), @varname(y)])

end
