module AutoGibbs

using Random: AbstractRNG
using Distributions
using DynamicPPL
import AbstractMCMC
import Distributions: logpdf

abstract type AbstractEvaluator end


struct SimpleModel{F} <: AbstractMCMC.AbstractModel
    f::F
end

(model::SimpleModel)(data...) = model.f(data...)


# sampling from prior: record all latent variables and their logpdf
struct SampleFromPrior <: AbstractEvaluator end

init_state(::SampleFromPrior) = (logp = Ref(0.0), latent = Dict{VarName, Tuple{<:Real, Float64}}())

function assume!(::SampleFromPrior, state, vn, dist)
    r = rand(dist)
    ℓ = logpdf(dist, r)
    push!(state.latent, vn => (r, ℓ))
    state.logp[] += ℓ
    return r, state
end

function observe!(::SampleFromPrior, state, vn, dist, obs)
    state.logp[] += logpdf(dist, obs)
    return state
end


# evaluataing the logpdf of a model given parameters θ
const VarDict{T} = Dict{VarName, T}

struct EvalLogp <: AbstractEvaluator
    θ::VarDict{<:Real}
end

init_state(::EvalLogp) = 0.0

function assume!(evaluator::EvalLogp, state, vn, dist)
    r = evaluator.θ[vn]
    state += logpdf(dist, r)
    return r, state
end

function observe!(evaluator::EvalLogp, state, vn, dist, obs)
    state += logpdf(dist, obs)
    return state
end

logpdf(model::SimpleModel, θ) = model(EvalLogp(θ))



# example sampler, like in https://turing.ml/dev/docs/using-turing/interface
# with static proposals
struct MetropolisHastings{D} <: AbstractMCMC.AbstractSampler
    init_θ::VarDict{<:Real}
    proposal::D
end

MetropolisHastings(init_θ::VarDict) =
    MetropolisHastings(init_θ, MvNormal(length(init_θ), 1))


# Create a very basic Transition type, only stores the 
# parameter draws and the log probability of the draw.
struct Transition{L}
    θ::VarDict{<:Real}
    logp::L
end

# Store the new draw and its log density.
Transition(model::SimpleModel, θ::VarDict) = Transition(θ, logpdf(model, θ))
logpdf(model::SimpleModel, t::Transition) = t.logp


function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::SimpleModel,
    spl::MetropolisHastings,
    N::Integer,
    ::Nothing;
    kwargs...
)
    return Transition(model, spl.init_θ)
end


# Define a function that makes a basic proposal.
function propose(rng, spl::MetropolisHastings, model::SimpleModel, θ::VarDict)
    # proposal = VarDict{Real}(keys(θ) .=> values(θ) .+ rand(rng, spl.proposal))
    proposal = VarDict{Real}(keys(θ) .=> rand(rng, spl.proposal))
    return Transition(model, proposal)
end
propose(rng, spl::MetropolisHastings, model::SimpleModel, t::Transition) =
    propose(rng, spl, model, t.θ)

# Calculates the probability `q(θ|θcond)`, using the proposal distribution `spl.proposal`.
q(spl::MetropolisHastings, θ::VarDict, θcond::VarDict) = logpdf(spl.proposal, values(θ) .- values(θcond))
q(spl::MetropolisHastings, t1::Transition, t2::Transition) = q(spl, t1.θ, t2.θ)


# Define the other step function. Returns a Transition containing
# either a new proposal (if accepted) or the previous proposal 
# (if not accepted).
function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::SimpleModel,
    spl::MetropolisHastings,
    ::Integer,
    t_prev::Transition;
    kwargs...
)
    # Generate a new proposal.
    t = propose(rng, spl, model, t_prev)

    # Calculate the log acceptance probability.
    log_α = logpdf(model, t) - logpdf(model, t_prev) + q(spl, t_prev, t) - q(spl, t, t_prev)

    # Decide whether to return the previous transition or the new one.
    if log(rand(rng)) < min(log_α, 0.0)
        return t
    else
        return t_prev
    end
end





# Example Model
# @model coinflip(y) = begin
#     p ~ Beta(1, 1)
#     N = length(y)
#     for i = 1:N
#         y[i] ~ Bernoulli(p)
#     end
# end

coinflip(y) = SimpleModel() do evaluator
    state = init_state(evaluator)
    p, state = assume!(evaluator, state, @varname(p), Beta(1, 1))

    N = length(y)
    for i = 1:N
        state = observe!(evaluator, state, @varname(y[i]), Bernoulli(p), y[i])
    end

    return state
end

model = coinflip([1,1,0])
println(model(SampleFromPrior()))
println(model(EvalLogp(VarDict{Real}(@varname(p) => 0.5))))
@show logpdf(Beta(1, 1), 0.5) + sum(logpdf.(Bernoulli(0.5), [1,1,0]))

# chain = AbstractMCMC.sample(model,
                            # MetropolisHastings(VarDict{Real}(@varname(p) => rand()),
                                               # Product([Uniform(0.3, 0.7)])),
                            # 10000)
# ps = [t.θ[@varname(p)] for t in chain]
# @show mean(ps)

end # module
