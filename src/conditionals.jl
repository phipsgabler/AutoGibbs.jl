using Distributions

"""
    conditioned(d0, blanket_logps)

Return a conditional distribution for the RV with distribution `d0` within a Markov blanket.

Constructed as

    P[X = x | conditioned] ∝ P[X = x | parents(X)] * P[children(X) | parents(children(X))]

equivalent to

    logpdf(D, x) = logpdf(d0, x) + ∑ blanket_logps

where the factors `blanket_logps` are the log probabilities in the Markov blanket.
"""
function conditioned(d0::DiscreteUnivariateDistribution, blanket_logps)
    Ω = support(d0)
    logblanket = reduce(+, blanket_logps; init=0.0)
    logtable = logpdf.(d0, Ω) .+ logblanket
    return DiscreteNonParametric(Ω, softmax(logtable))
end


# from https://github.com/JuliaStats/StatsFuns.jl/blob/master/src/basicfuns.jl#L259
function softmax(x::AbstractArray)
    n = length(x)
    u = maximum(x)
    s = 0.0
    r = similar(x, Float64)
    
    @inbounds for i = 1:n
        s += (r[i] = exp(x[i] - u))
    end
    
    invs = inv(s)
    
    @inbounds for i = 1:n
        r[i] *= invs
    end
    
    return r
end



# D_w = Dirichlet(2, 1.0)
# w = rand(D_w)
# D_p = DiscreteNonParametric([0.3, 0.6], w)
# p = rand(D_p)
# D_x = Bernoulli(p)
# x = rand(D_x)
# conditioned(D_p, [logpdf(D_x, x)])
