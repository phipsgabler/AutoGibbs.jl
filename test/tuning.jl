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

function evaluate(model::String, samplers::String, stepsize::String, steps::String, particles::String)
    return evaluate(
        model,
        samplers,
        parse(Float64, stepsize),
        parse(Int, steps),
        parse(Int, particles)
    )
end

"""
    evaluate(model, samplers, stepsize, steps, particles)

Try out the model with specified samplers and parameters.  `stepsize` and `steps` are for HMC,
`particles` is for PG.  `samplers` is a string containing either `nuts` or `hmc`, and either
`ag` or `pg`.  Unneeded parameters are ignored, but must be present for the signature.
"""
function evaluate(model, samplers, stepsize, steps, particles)
    generate, example, tarray_example, p_discrete, p_continuous = MODEL_SETUPS[model]

    dataname, data = generate(DATA_RNG, DATA_SIZE)
    model_pg = tarray_example(; dataname => data)

    if occursin(r"nuts"i, samplers)
        println("NUTS for continuous sampler.")
        dummy_nuts = NUTS()
        ad, _, metric = typeof(dummy_nuts).parameters
        sampler_continuous = NUTS{ad, (p_continuous...,), metric}(
            map(f -> getfield(dummy_nuts, f), fieldnames(typeof(dummy_nuts)))...
        )
    elseif occursin(r"hmc"i, samplers)
        println("HMC for continuous sampler.")
        sampler_continuous = HMC(stepsize, steps, p_continuous...)
    else
        error("Unknown continuous sampler")
    end
    
    if occursin(r"ag"i, samplers)
        println("AG for discrete sampler. Compiling static conditional...")
        sampler_discrete = StaticConditional(model_pg, p_discrete)
        println("Done.")
    elseif occursin(r"pg"i, samplers)
        println("PG for discrete sampler.")
        sampler_discrete = PG(particles, p_discrete)
    else
        error("Unknown discrete sampler")
    end
    
    @show data
    chain = sample(model_pg, Gibbs(sampler_continuous, sampler_discrete), 1000)
    show(chain)
    plot(chain)
end
