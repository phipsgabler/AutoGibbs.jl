using Turing
using Turing.RandomMeasures
using DynamicPPL
using Random
using AutoGibbs
using Plots
using StatsPlots
using ProgressMeter
using Dates
using InteractiveUtils


include("models.jl")


const CHAIN_LENGTH = 10 #5_000   # sampling steps
const HMC_LF_SIZE = 0.1      # parameter 1 for HMC
const HMC_N_STEP = 10        # parameter 2 for HMC
const BENCHMARK_CHAINS = 10  # number of chains to sample per combination
const DATA_SIZES = (5,)# (10, 25, 50, 100)
const N_PARTICLES = (5,)#(5, 10, 15)
const DATA_RNG = MersenneTwister(424242)


function run_experiments(
    modelname,
    generate,
    example,
    tarray_example,
    p_discrete,
    p_continuous,
    compilation_times_channel,
    chains_channel
)
    old_prog = Turing.PROGRESS[]
    Turing.turnprogress(false)
   
    for L in DATA_SIZES
        data = generate(DATA_RNG, L)
        model_ag = example(x = data)
        model_pg = tarray_example(x = data)

        for (i, particles) in enumerate(N_PARTICLES)
            # get a new conditional for each particle size, so that we have
            # a couple of samples of the compilation time for each L
            start_time = time_ns()
            static_conditional = StaticConditional(model_ag, p_discrete)
            compilation_time = (time_ns() - start_time) / 1e9
            @info "Compiled conditional for data size $L in $compilation_time seconds"
            
            put!(compilation_times_channel,
                  (model = modelname,
                   data_size = L,
                   repetition = i,
                   compilation_time = compilation_time))
            
            # mh = MH(p_continuous...)
            hmc = HMC(HMC_LF_SIZE, HMC_N_STEP, p_continuous...)
            pg = PG(particles, p_discrete)

            combinations = Dict([
                # ("AG", "MH") => (model_ag, Gibbs(static_conditional, mh)),
                ("AG", "HMC") => (model_ag, Gibbs(static_conditional, hmc)),
                # ("PG", "MH") => (model_pg, Gibbs(pg, mh)),
                ("PG", "HMC") => (model_pg, Gibbs(pg, hmc))
            ])

            for ((d_alg, c_alg), (model, sampler)) in combinations
                @info "Sampling $BENCHMARK_CHAINS chains using $d_alg+$c_alg with data size $L and $particles particles"
                progress = Progress(BENCHMARK_CHAINS)
                
                Threads.@threads for n in 1:BENCHMARK_CHAINS
                    start_time = time_ns()
                    chain = sample(model, sampler, CHAIN_LENGTH)
                    # sample(model, sampler, MCMCThreads(), 10, N)
                    sampling_time = (time_ns() - start_time) / 1e9
                    
                    put!(chains_channel,
                         (model = modelname,
                          discrete_algorithm = d_alg,
                          continuous_algorithm = c_alg,
                          particles = particles,
                          data_size = L,
                          repetition = n,
                          sampling_time = sampling_time,
                          chain = chain,
                          ))
                    
                    next!(progress)
                end
            end
        end
    end

    Turing.turnprogress(old_prog)

    close(compilation_times_channel)
    close(chains_channel)
end


VARIABLE_NAMES = (
    "model",
    "discrete_algorithm",
    "continuous_algorithm",
    "particles",
    "data_size",
    "repetition"
)

function print_csv_line(io, values...)
    join(io, ("\"$(escape_string(string(v)))\"" for v in values), ",")
end

function serialize_chains(filename_times, filename_diagnostics, filename_chains, observations)
    open(filename_times, "w") do f_times
        open(filename_diagnostics, "w") do f_diag
            open(filename_chains, "w") do f_chains
                # write header lines
                print_csv_line(f_times, VARIABLE_NAMES..., "sampling_time")
                print_csv_line(f_diag, VARIABLE_NAMES..., "parameter", "diagnostic", "value")
                print_csv_line(f_chains, VARIABLE_NAMES..., "parameter", "step", "value")
                
                for obs in observations
                    condition_data = Tuple(obs)[1:end-2]
                    param_names = names(obs.chain, :parameters)

                    # write line for the sampling time to `f_times`
                    print(f_times, '\n')
                    print_csv_line(f_times, condition_data..., obs.sampling_time)

                    # write line for parameter and diagnostic per chain to `f_diag`
                    diagnostics = ess(obs.chain)
                    for p in param_names, d in (:ess, :r_hat)
                        print(f_diag, '\n')
                        print_csv_line(f_diag, condition_data..., p, d, diagnostics[p, d])
                    end
                    
                    # write line for each parameter and value in the chain to `f_chains`
                    for (step, sampling) in enumerate(eachrow(Array(obs.chain)))
                        for (p, value) in zip(param_names, sampling)
                            print(f_chains, '\n')
                            print_csv_line(f_chains, condition_data..., p, step, value)
                        end
                    end
                end
            end
        end
    end
end

function serialize_compilation_times(filename, observations)
    open(filename, "w") do f
        print_csv_line(f, "model", "data_size", "repetition", "compilation_time")

        for obs in observations
            print(f, '\n')
            print_csv_line(f, obs...)
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

    timestamp = round(Dates.now(), Dates.Minute)
    samplingtimes_fn = abspath(results_path, "$modelname-sampling_times-$timestamp.csv")
    diagnostics_fn = abspath(results_path, "$modelname-diagnostics-$timestamp.csv")
    chains_fn = abspath(results_path, "$modelname-chains-$timestamp.csv")
    compiletimes_fn = abspath(results_path, "$modelname-compile_times-$timestamp.csv")

    samplingtimes_fn_nodate = abspath(results_path, "$modelname-sampling_times.csv")
    diagnostics_fn_nodate = abspath(results_path, "$modelname-diagnostics.csv")
    chains_fn_nodate = abspath(results_path, "$modelname-chains.csv")
    compiletimes_fn_nodate = abspath(results_path, "$modelname-compile_times.csv")
    
    if haskey(MODEL_SETUPS, modelname)
        compilation_times_channel = Channel()
        chains_channel = Channel()
        
        @sync begin
            @async run_experiments(modelname, MODEL_SETUPS[modelname]...,
                                   compilation_times_channel,
                                   chains_channel)
            @async serialize_chains(samplingtimes_fn, diagnostics_fn, chains_fn, chains_channel)
            @async serialize_compilation_times(compiletimes_fn, compilation_times_channel)
        end

        # overwrite a symlink to the latest version, without time stamp
        for (file, link) in zip(
            (samplingtimes_fn, diagnostics_fn, chains_fn, compiletimes_fn),
            (samplingtimes_fn_nodate, diagnostics_fn_nodate, chains_fn_nodate, compiletimes_fn_nodate)
        )
            rm(link, force=true)
            symlink(file, link)
        end
    else
        println("Unknown model: $modelname")
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    println("Using $(Threads.nthreads()) threads for parallel test chains.")
    main(ARGS...)

    println("Executed on:")
    InteractiveUtils.versioninfo()
end
