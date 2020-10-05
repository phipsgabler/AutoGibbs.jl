using Turing
using Turing.RandomMeasures
using Random
using AutoGibbs
using BenchmarkTools
using Plots
using StatsPlots


include("models.jl")
include("models_comparison.jl")


function benchmark_models(combinations...; N=100, bparams...)
    old_prog = Turing.PROGRESS[]
    Turing.turnprogress(false)
    suite = BenchmarkGroup()
    
    benchmarks = map(combinations) do (name, (m, alg))
        suite[name] = @benchmarkable sample($m, $alg, $N)
    end
    
    tune!(suite)
    results = run(suite;bparams...)
    
    Turing.turnprogress(old_prog)
    return results
end

function compare_gmm()
    data = gmm_generate(20)
    m1 = gmm_example(x = data)
    m2 = gmm_tarray_example(x = data)

    cond_gmm_z = StaticConditional(m1, :z)
    alg1 = Gibbs(cond_gmm_z, MH(:w, :μ))
    alg2 = Gibbs(cond_gmm_z, HMC(0.01, 10, :w, :μ))
    alg3 = Gibbs(PG(20, :z), MH(:w, :μ))
    alg4 = Gibbs(PG(20, :z), HMC(0.01, 10, :w, :μ))

    b = benchmark_models("AG+MH" => (m1, alg1),
                         "AG+HMC" => (m1, alg2),
                         "PG+MH" => (m2, alg3),
                         "PG+HMC" => (m2, alg4);
                         N=1000, verbose=true, samples=10)
    display(b)
end


compare_gmm()
