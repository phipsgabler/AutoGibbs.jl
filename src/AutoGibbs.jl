module AutoGibbs

using Distributions
using DynamicPPL
import AbstractMCMC

abstract type AbstractEvaluator end


struct SimpleModel{F} <: AbstractMCMC.AbstractModel
    f::F
end

(model::SimpleModel)(data...) = function evaluate(evaluator::AbstractEvaluator)
    return model.f(evaluator, data...)
end


# sampling from prior: record all latent variables and their logpdf
struct Sample <: AbstractEvaluator end

init_state(::Sample) = (logp = Ref(0.0), latent = Dict{VarName, Tuple{<:Real, Float64}}())

function assume!(::Sample, state, vn, dist)
    r = rand(dist)
    ℓ = logpdf(dist, r)
    push!(state.latent, vn => (r, ℓ))
    state.logp[] += ℓ
    return r, state
end

function observe!(::Sample, state, vn, dist, obs)
    state.logp[] += logpdf(dist, obs)
    return state
end


const coinflip = SimpleModel() do evaluator, y
    state = init_state(evaluator)
    θ, state = assume!(evaluator, state, @varname(θ), Beta(1, 1))

    N = length(y)
    for i = 1:N
        state = observe!(evaluator, state, @varname(y[i]), Bernoulli(θ), y[i])
    end

    return state
end

model = coinflip([1,1,0])
println(model(Sample()))





end # module
