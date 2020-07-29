using DataStructures: DefaultDict
using Distributions
using DynamicPPL


export conditional_dists


"""
Fix all except the `N`th argument of `D`, and the observed value; if 

    ℓ = LogLikelihood{N}(D, value, args...)

then

    ℓ(x) = logpdf(D(args[1], ..., args[N-1], x, args[N+1], ..., args[K]), value)
"""
struct LogLikelihood{N, D<:Distribution, T, Args<:Tuple}
    value::T
    args::Args
    
    function LogLikelihood{N}(::Type{D}, value, args...) where {N, D<:Distribution}
        L = length(args) + 1
        (1 ≤ N ≤ L) || throw(ArgumentError("$N is not in 1:$L"))
        return new{N, D, typeof(value), typeof(args)}(value, args)
    end
end

function Base.show(io::IO, ℓ::LogLikelihood{N, D}) where {N, D}
    L = length(ℓ.args) + 1
    print(io, "θ -> logpdf(", D, "(")
    join(io, _splice(N, "θ", ℓ.args), ", ")
    print(io, "), ", ℓ.value, ")")
end

function (ℓ::LogLikelihood{N, D, T, Args})(θ) where {N, D, T, Args}
    return logpdf(D(_splice(N, θ, ℓ.args)...), ℓ.value)
end


"""
    _splice(position, value, args)

Insert `value` into `args` at `position`: `_splice(2, :x, (1, 2, 3)) ~> (1, :x, 2, 3)`.
"""
function _splice(position, value, args::Tuple)
    if position == 1
        return (value, args...)
    elseif 1 < position ≤ length(args) + 1
        return (first(args), _splice(position - 1, value, Base.tail(args))...)
    else
        throw(ArgumentError("Can't insert value into $args at position $position"))
    end
end


struct Conditional{D<:Distribution, B}
    var_dist::D
    blanket_dists::B
end

function (cond::Conditional{V})(blanket) where {V}
    blanket_logp = sum(cond.blanket_dists[b](blanket[b]) for b in keys(cond.blanket_dists))
    return conditioned(cond.dist, blanket_logp)
end


components(vn, indices) = [VarName(getsym(vn), ix) for ix in indices]

"""
    conditional_dists(graph, varname)

Derive a dictionary of Gibbs conditionals for all assumption statements in `graph` that are subsumed
by `varname`.
"""
function conditional_dists(graph, varname)
    # There can be multiple tildes for one `varname`, e.g., `x[1], x[2]` both subsumed by `x`.
    dists = Dict{VarName, Distribution}()
    blankets = DefaultDict{VarName, Float64}(0.0)
    
    for (ref, stmt) in graph
        # record distribution of every matching node
        if !isnothing(ref.vn)
            ref_vn = dereference(graph, ref.vn)
            
            if DynamicPPL.subsumes(varname, ref_vn)
                dist = getvalue(stmt.dist)
                dists[ref_vn] = dist

                # if dist isa Product
                #     components = dist.v
                #      for i in eachindex(components)
                #          dists[VarName(ref_vn, ((i,),))] = components[i]
                #      end
                # end
            end

            # update the blanket logp for all matching parents
            for p in dependencies(stmt.dist)
                # !isnothing(p.vn) && @show dereference(graph, p.vn)
                if !isnothing(p.vn) && any(DynamicPPL.subsumes(r, dereference(graph, p.vn))
                                           for r in keys(dists))
                    # @show p
                    child_dist, child_value = getvalue(stmt.dist), try_getvalue(graph, stmt.value)
                    ℓ = logpdf(child_dist, child_value)
                    # @show ℓ
                    blankets[dereference(graph, p.vn)] += ℓ
                end
            end
        end
    end

    # @show blankets
    @show keys(dists)
    return Dict(vn => conditioned(d, blankets[vn]) for (vn, d) in dists)
end


# function conditionals(graph, varname)
#     # There can be multiple tildes for one `varname`, e.g., `x[1], x[2]` both subsumed by `x`.
#     dists = Dict{Reference, Distribution}()
#     blankets = DefaultDict{Reference, Vector{LogLikelihood}}(Vector{LogLikelihood})
    
#     for (ref, stmt) in graph
#         # record distribution of every matching node
#         if !isnothing(ref.vn) && DynamicPPL.subsumes(varname, ref.vn)
#             dists[ref] = getvalue(stmt.dist)
#         end

        
#         for p in parents(stmt)
#             if p isa Union{Assumption, Observation} && haskey(dists, p)
                
#             end
#         end
        
#         # record all parents that are random variables
#         direct_dependencies = Reference[arg for arg in stmt.args if arg isa Reference]
#         mapreduce(dependencies,
#                          append!,
#                          direct_dependencies,
#                          init=direct_dependencies)
#         rvs = filter((i, p) -> haskey(dists, p), dependencies(stmt))

#         # update the blanket likelihoods for all rv parents

#         for (i, p) in rvs
#             child = graph[p]
#             child_dist, child_value = getvalue(child.dist), child.value
#             push!(blankets[p], LogLikelihood{i}(typeof(child_dist), child_value, ))
#         end
#     end

#     return Dict(dereference(graph, r.vn) => nothing for (r, d) in dists)
# end



DynamicPPL.getlogp(tilde::Union{Assumption, Observation}) = logpdf(tilde.dist, tilde.value)



"""
    conditioned(d0, blanket_logps)

Return an array of distributions for the RV with distribution `d0` within a Markov blanket.

Constructed as

    P[X = x | conditioned] ∝ P[X = x | parents(X)] * P[children(X) | parents(children(X))]

equivalent to

    logpdf(D, x) = logpdf(d0, x) + ∑ blanket_logps

where the factors `blanket_logps` are the log probabilities in the Markov blanket.

The result is an array to allow to condition `Product` distributions.
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
    
    s⁻¹ = inv(s)
    
    @inbounds for i in eachindex(x)
        x[i] *= s⁻¹
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
