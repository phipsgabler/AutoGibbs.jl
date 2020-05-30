using Distributions
using DynamicPPL


function conditional_dists(graph, varname)
    subsumed = findall(graph, r -> !isnothing(r.vn) && DynamicPPL.subsumes(varname, r.vn), keys(graph))
    dists = similar(subsumed, DiscreteNonParametric)
    
    for n in eachindex(subsumed, dists)
        noderef = subsumed[n]
        dist = getdist(graph, noderef)
        blanket = getblanket(graph, noderef)
        dists[n] = conditioned(dist, blanket)
    end

    return dists
end

getdist(graph, noderef) = getvalue(graph[noderef].dist)

function getblanket(graph, noderef)
    children = filter(graph, )
end


DynamicPPL.getlogp(tilde::Union{Assumption, Observation}) = logpdf(tilde.dist, tilde.value)

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
    try
        Ω = support(d0)
    catch
        throw(ArgumentError("Unable to get the support of $d0 (probably infinite)"))
    end
    
    logblanket = reduce(+, blanket_logps; init=0.0)
    logtable = logpdf.(d0, Ω) .+ logblanket
    return DiscreteNonParametric(Ω, softmax!(logtable))
end

conditioned(::Distribution, blanket_logps) =
    throw(ArgumentError("Cannot condition a non-discrete or non-univariate distribution."))


# from https://github.com/JuliaStats/StatsFuns.jl/blob/master/src/basicfuns.jl#L259
function softmax!(x::AbstractArray{<:AbstractFloat})
    u = maximum(x)
    s = zero(u)
    
    @inbounds for i in eachindex(x)
        s += (x[i] = exp(x[i] - u))
    end
    
    invs = inv(s)
    
    @inbounds for i in eachindex(x)
        x[i] *= invs
    end
    
    return x
end



# D_w = Dirichlet(2, 1.0)
# w = rand(D_w)
# D_p = DiscreteNonParametric([0.3, 0.6], w)
# p = rand(D_p)
# D_x = Bernoulli(p)
# x = rand(D_x)
# conditioned(D_p, [logpdf(D_x, x)])
