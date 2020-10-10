using Turing
using Turing.RandomMeasures
using DynamicPPL
using Random
using AutoGibbs
using Plots
using StatsPlots
using ProgressMeter


include("models.jl")


const CHAIN_LENGTH = 10 #5_000    # sampling steps
const HMC_LF_SIZE = 0.1   # parameter 1 for HMC
const HMC_N_STEP = 10     # parameter 2 for HMC
const BENCHMARK_CHAINS = 10 # number of chains to sample per combination
const DATA_RNG = MersenneTwister(424242)


function run_experiments(
    modelname,
    generate,
    example,
    tarray_example,
    p_discrete,
    p_continuous
)
    chains = []
    compilation_times = []

    old_prog = Turing.PROGRESS[]
    Turing.turnprogress(false)
   
    for L in (10,)#(10, 25, 50)
        data = generate(DATA_RNG, L)
        model_ag = example(x = data)
        model_pg = tarray_example(x = data)

        for particles in (5,)#(5, 10, 15)
            # get a new conditional for each particle size, so that we have
            # a couple of samples of the compilation time for each L
            start_time = time_ns()
            static_conditional = StaticConditional(model_ag, p_discrete)
            compilation_time = (time_ns() - start_time) / 1e9
            @info "Compiled conditional in $compilation_time seconds"
            
            push!(compilation_times,
                  (model = modelname,
                   data_size = L,
                   compilation_time = compilation_time))
            
            mh = MH(p_continuous...)
            hmc = HMC(HMC_LF_SIZE, HMC_N_STEP, p_continuous...)
            pg = PG(particles, p_discrete)

            combinations = Dict([
                # ("AG", "MH") => (model_ag, Gibbs(static_conditional, mh)),
                # ("AG", "HMC") => (model_ag, Gibbs(static_conditional, hmc)),
                ("PG", "MH") => (model_pg, Gibbs(pg, mh)),
                ("PG", "HMC") => (model_pg, Gibbs(pg, hmc))
            ])

            for ((d_alg, c_alg), (model, sampler)) in combinations
                @info "Sampling $BENCHMARK_CHAINS chains using $d_alg+$c_alg with data size $L and $particles particles"
                @showprogress for n in 1:BENCHMARK_CHAINS
                    start_time = time_ns()
                    chain = sample(model, sampler, CHAIN_LENGTH)
                    # sample(model, sampler, MCMCThreads(), 10, N)
                    sampling_time = (time_ns() - start_time) / 1e9
                    
                    push!(chains,
                          (model = modelname,
                           discrete_algorithm = d_alg,
                           continuous_algorithm = c_alg,
                           particles = particles,
                           data_size = L,
                           repetition = n,
                           sampling_time = sampling_time,
                           chain = chain,
                           ))
                end
            end
        end
    end

    Turing.turnprogress(old_prog)

    return chains, compilation_times
end


VARIABLE_NAMES = (
    "model",
    "discrete_algorithm",
    "continuous_algorithm",
    "particles",
    "data_size",
    "repetition"
)

function serialize_chains(filename_times, filename_chains, observations)
    open(filename_times, "w") do f_times
        open(filename_chains, "w") do f_chains
            join(f_times, VARIABLE_NAMES, ",")
            print(f_times, ",sampling_time")
            
            join(f_chains, VARIABLE_NAMES, ",")
            print(f_chains, ",parameter,step,value")
            
            for obs in observations
                condition_data = Tuple(obs)[1:end-2]

                # write line for the sampling time
                print(f_times, '\n')
                join(f_times, condition_data, ",")
                print(f_times, ",", obs.sampling_time)

                # write line for each parameter and value in the chain
                param_names = names(obs.chain, :parameters)
                for (step, sampling) in enumerate(eachrow(Array(obs.chain)))
                    for (p, value) in zip(param_names, sampling)
                        print(f_chains, '\n')
                        join(f_chains, condition_data, ",")
                        print(f_chains, ",", p, ",", step, ",", value)
                    end
                end
            end
        end
    end
end

function serialize_compilation_times(filename, observations)
    open(filename, "w") do f
        print(f, "model,data_size,sampling_time")

        for obs in observations
            print(f, '\n')
            join(f, obs, ",")
        end
    end
end



const MODEL_SETUPS = Dict([
    "GMM" => (gmm_generate, gmm_example, gmm_tarray_example, :z, (:w, :μ)),
    "HMM" => (hmm_generate, hmm_example, hmm_tarray_example, :s, (:t, :m)),
    "IMM" => (imm_stick_generate, imm_stick_example, imm_stick_tarray_example, :z, (:μ, :v))
])

function main(modelname, results_path=nothing)
    modelname = uppercase(modelname)
    
    if isnothing(results_path)
        results_path = "."
    end

    samplingtimes_fn = joinpath(results_path, "sampling_times.csv")
    chains_fn = joinpath(results_path, "chains.csv")
    compiletimes_fn = joinpath(results_path, "compile_times.csv")
    
    if haskey(MODEL_SETUPS, modelname)
        chains, compilation_times = run_experiments(modelname, MODEL_SETUPS[modelname]...)
        serialize_chains(samplingtimes_fn, chains_fn, chains)
        serialize_compilation_times(compiletimes_fn, compilation_times)
    else
        println("Unknown model: $modelname")
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
