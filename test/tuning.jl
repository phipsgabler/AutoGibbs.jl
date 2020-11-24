using Turing
using Turing.RandomMeasures
using DynamicPPL
using Random
using AutoGibbs
using StatsPlots

include("models.jl")


const DATA_RNG = MersenneTwister(424242)
const MODEL_SETUPS = Dict([
    "GMM" => (gmm_generate, gmm_example, gmm_tarray_example, :z, (:w, :μ)),
    "HMM" => (hmm_generate, hmm_example, hmm_tarray_example, :s, (:t, :m)),
    "IMM" => (imm_stick_generate, imm_stick_example, imm_stick_tarray_example, :z, (:μ, :v))
])
const DATA_SIZE = 10

function evaluate(model::String, nuts::String, stepsize::String, steps::String, particles::String)
    return evaluate(
        model,
        parse(Bool, nuts),
        parse(Float64, stepsize),
        parse(Int, steps),
        parse(Int, particles)
    )
end

function evaluate(model, nuts, stepsize, steps, particles)
    generate, example, tarray_example, p_discrete, p_continuous = MODEL_SETUPS[model]

    dataname, data = generate(DATA_RNG, DATA_SIZE)
    model_pg = tarray_example(; dataname => data)

    if nuts
        dummy_nuts = NUTS()
        ad, _, metric = typeof(dummy_nuts).parameters
        sampler = NUTS{ad, (p_continuous...,), metric}(
            map(f -> getfield(dummy_nuts, f), fieldnames(typeof(n)))...
        )
    else
        sampler = HMC(stepsize, steps, p_continuous...)
    end
    
    @show data
    chain = sample(model_pg, Gibbs(sampler, PG(particles, p_discrete)), 1000)
    show(chain)
    plot(chain)
end
