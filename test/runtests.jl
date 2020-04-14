using AutoGibbs
using Test

using Distributions


@testset "AutoGibbs.jl" begin
    (α_0, θ_0) = (2.0, inv(3.0))
    @model function inverse_gdemo(x)
        λ ~ Gamma(α_0, θ_0)
        m ~ Normal(0, sqrt(1 / λ))
        x ~ Normal(m, sqrt(1 / λ))
    end

    model = inverse_gdemo(1.4)
    st = AutoGibbs.strip_calls(trackmodel(model))


    @model function test1(x)
        s ~ Gamma(1.0, 1.0)
        x .~ Normal(0.0, s)
    end

    t1 = trackmodel(test1([-0.5, 0.5])) |> AutoGibbs.strip_calls
    d1 = dependencies(t1)
    
    @model function test2(x)
        s ~ Gamma(1.0, 1.0)
        for i in eachindex(x)
            x[i] ~ Normal(float(i), s)
        end
    end

    t2 = trackmodel(test2([-0.5, 0.5])) |> AutoGibbs.strip_calls
    d2 = dependencies(t2)
    
    @model function test3(w)
        p ~ Beta(1, 1)
        x ~ Bernoulli(p)
        y ~ Bernoulli(p)
        z ~ Bernoulli(p)
        w ~ MvNormal([x, y, z], 1.2)
    end

    t3 = trackmodel(test3([1, 0, 1])) |> AutoGibbs.strip_calls
    d3 = dependencies(t3)

    @model function mixture(x) 
        μ ~ MvNormal(fill(0, 2), 2.0)
        z = Vector{Int}(undef, length(x))
        for i in 1:length(x)
            z[i] ~ Categorical([0.5, 0.5])
            x[i] ~ Normal(μ[z[i]], 0.1)
        end
    end
end
