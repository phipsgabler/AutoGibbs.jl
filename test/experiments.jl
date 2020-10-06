using Turing
using Turing.RandomMeasures
using Random
using AutoGibbs
using BenchmarkTools
using Plots
using StatsPlots


include("models.jl")
include("models_comparison.jl")

const plots_dir = "plots"

const iter = 5_000    # sampling steps
const N = 20          # size of test data set
const particles = 10  # parameter for particle gibbs
const lf_size = 0.1   # parameter 1 for HMC
const n_step = 10     # parameter 2 for HMC


function benchmark_models(combinations...; N=iter, bparams...)
    old_prog = Turing.PROGRESS[]
    Turing.turnprogress(false)
    suite = BenchmarkGroup()
    
    benchmarks = map(combinations) do (name, (m, alg))
        suite[name] = @benchmarkable sample($m, $alg, $N)
    end
    
    tune!(suite)
    results = run(suite; bparams...)
    
    Turing.turnprogress(old_prog)
    return results
end


function compare_samplers(model_ag, model_pg, p_discrete, p_continuous)
    static_conditional = StaticConditional(model_ag, p_discrete...)
    alg1 = Gibbs(static_conditional, MH(p_continuous...))
    alg2 = Gibbs(static_conditional, HMC(lf_size, n_step, p_continuous...))
    alg3 = Gibbs(PG(particles, p_discrete...), MH(p_continuous...))
    alg4 = Gibbs(PG(particles, p_discrete...), HMC(lf_size, n_step, p_continuous...))

    combinations = Dict([
        "AG+MH" => (model_ag, alg1),
        "AG+HMC" => (model_ag, alg2),
        "PG+MH" => (model_pg, alg3),
        "PG+HMC" => (model_pg, alg4)])
    
    # b = benchmark_models(combinations...;
                         # N=iter, verbose=true, samples=10)

    pd = mkpath(plots_dir)
    for (name, (m, alg)) in combinations
        chain = sample(m, alg, iter)
        # plot only the first of the cluster paramers, and all continuous ones
        plt = plot(chain[[Symbol.(p_discrete, Ref("[1]"))..., p_continuous...]],
                   title=name)
        Plots.pdf(plt, joinpath(pd, name * ".pdf"))
    end
    
    # return b
end

function compare_gmm()
    data = gmm_generate(20)
    m1 = gmm_example(x = data)
    m2 = gmm_tarray_example(x = data)
    p_discrete = [:z]
    p_continuous = [:w, :μ]

    return compare_samplers(m1, m2, p_discrete, p_continuous)
end

function compare_hmm()
    data = hmm_generate(20)
    m1 = hmm_example(x = data)
    m2 = hmm_tarray_example(x = data)
    p_discrete = [:s]
    p_continuous = [:t, :m]

    return compare_samplers(m1, m2, p_discrete, p_continuous)
end

function compare_imm()
    m1 = imm_stick_example()
    m2 = imm_stick_tarray_example()
    p_discrete = [:z]
    p_continuous = [:μ, :v]

    return compare_samplers(m1, m2, p_discrete, p_continuous)
end



compare_gmm()
# compare_hmm()
# compaare_imm()
