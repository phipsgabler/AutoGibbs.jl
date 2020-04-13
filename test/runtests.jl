using AutoGibbs
using Test

using Distributions


@testset "AutoGibbs.jl" begin
    (α_0, θ_0) = (2.0, inv(3.0))
    @model inverse_gdemo(x) = begin
        λ ~ Gamma(α_0, θ_0)
        m ~ Normal(0, sqrt(1 / λ))
        x ~ Normal(m, sqrt(1 / λ))
    end

    model = inverse_gdemo(1.4)
    st = AutoGibbs.strip_calls(trackmodel(model))
end
