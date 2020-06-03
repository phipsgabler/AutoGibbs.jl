using DataStructures: DefaultDict
using Distributions
using DynamicPPL


function conditional_dists(graph, varname)
    dists = Dict{Reference, Distribution}()
    blankets = DefaultDict{Reference, Float64}(0.0)
    
    for (ref, stmt) in graph
        # record distribution of every matching node
        if !isnothing(ref.vn) && DynamicPPL.subsumes(varname, ref.vn)
            dists[ref] = getvalue(stmt.dist)
        end

        # update the blanket logp for all matching parents
        for p in dependencies(stmt)
            if haskey(dists, p)
                child = graph[p]
                child_dist, child_value = getvalue(child.dist), child.value
                blankets[p] += logpdf(child_dist, child_value)
            end
        end
    end

    return Dict([r => conditioned(d, blankets[r]) for (r, d) in dists])
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
function conditioned(d0::DiscreteUnivariateDistribution, blanket_logp)
    local Ω

    try
        Ω = support(d0)
    catch
        throw(ArgumentError("Unable to get the support of $d0 (probably infinite)"))
    end
    
    logtable = logpdf.(d0, Ω) .+ blanket_logp
    return DiscreteNonParametric(Ω, softmax!(logtable))
end

# `Product`s can be treated as an array of iid variables
conditioned(d0::Product, blanket_logp) = Product(conditioned.(d0.v, blanket_logp))

conditioned(d0::Distribution, blanket_logps) =
    throw(ArgumentError("Cannot condition a non-discrete or non-univariate distribution $d0."))


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
