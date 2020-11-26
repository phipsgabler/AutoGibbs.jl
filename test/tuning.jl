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
const CHAIN_LENGTH = 500

function evaluate_graphic(
    model::String,
    samplers::String,
    stepsize::String,
    steps::String,
    particles::String
)
    data, chain = evaluate(
        model,
        samplers,
        parse(Float64, stepsize),
        parse(Int, steps),
        parse(Int, particles)
    )
    @show data
    @show chain
    plot(chain)
end

function evaluate_grid(
    model,
    samplers;
    stepsizes=(0.01, 0.025, 0.05, 0.1),
    steps=(5, 10),
    particles=(20, 40, 60),
    N=CHAIN_LENGTH)

    for ss in stepsizes, s in steps, p in particles
        println("Evaluating $model with $samplers")
        println("using stepsize $ss, $s steps, and $p particles")
        println("for chain of length $N")
        data, chain = evaluate(model, samplers, ss, s, p, N)
        show(ess(chain))
    end
end

function evaluate_marginalized(
    stepsizes=(0.01, 0.025, 0.05, 0.1),
    steps=(5, 10),
    N=CHAIN_LENGTH
)
    dataname, data = gmm_marginalized_generate(DATA_RNG, 10)
    model = gmm_marginalized_example(; dataname => data)
    @show model
    for ss in stepsizes, s in steps
        println("Using stepsize $ss, $s steps, for chain of length $N")
        chain = sample(model, HMC(ss, s), N)
        show(ess(chain))
    end
end

"""
    evaluate(model, samplers, stepsize, steps, particles)

Try out the model with specified samplers and parameters.  `stepsize` and `steps` are for HMC,
`particles` is for PG.  `samplers` is a string containing either `nuts` or `hmc`, and either
`ag` or `pg`.  Unneeded parameters are ignored, but must be present for the signature.

Example:
```julia
evaluate_grid("GMM", "ag,nuts", stepsizes=(nothing,), steps=(nothing,))
```
"""
function evaluate(model, samplers, stepsize, steps, particles, N)
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
    
    chain = sample(model_pg, Gibbs(sampler_continuous, sampler_discrete), N)
    return data, chain
end
