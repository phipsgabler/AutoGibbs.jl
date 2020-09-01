using DataStructures: DefaultDict
using Distributions
using DynamicPPL
using Turing.RandomMeasures


export conditionals, sampled_values


"""
    abstract type Cont end

`Cont` is an analytic representation of joint distribution represented in a `Graph` -- a function
from a dictionary of variable assignments to a log probability.  (For lack of a better term, I call
these functions "continuations".)

Each `Call` gets associated with a `Transformation`, and each tilde statement with a
`LogLikelihood`.  Those are both callable with a dict argument.  SSA variables get transformed to
either a `Fixed` value for constants, a `Variable` for assignments of random variables, or the
`Cont` they come from.

Something like this:

```
⟨28⟩ => logpdf(Normal(), θ[μ[2]])
⟨29⟩ => getindex(⟨8⟩(array initializer with undefined values, 2), getfield(iterate(Colon()(1, 2), getfield(iterate(Colon()(1, 2)), 2)), 1))
⟨31⟩ => logpdf(Dirichlet(2, 1.0), θ[w])
⟨32⟩ => Array{Int64,1}(array initializer with undefined values, length([0.1, -0.05, 1.0]))
⟨33⟩ => Colon()(1, length([0.1, -0.05, 1.0]))
⟨34⟩ => iterate(Colon()(1, length([0.1, -0.05, 1.0])))
⟨36⟩ => getfield(iterate(Colon()(1, length([0.1, -0.05, 1.0]))), 1)
⟨37⟩ => getfield(iterate(Colon()(1, length([0.1, -0.05, 1.0]))), 2)
⟨42⟩ => logpdf(DiscreteNonParametric(θ[w]), θ[z[1]])
```

for 

```
⟨28⟩ = μ[2] ~ Normal() → -1.2107564627453093
⟨29⟩ = μ[2] = getindex(⟨9⟩, ⟨23⟩) → -1.2107564627453093
⟨31⟩ = w ~ Dirichlet(⟨4⟩, 1.0) → [0.7023731332410442, 0.2976268667589558]
⟨32⟩ = Array{Int64,1}(array initializer with undefined values, ⟨7⟩) → [139815315085536, 139815315085552, 139815315085568]
⟨33⟩ = Colon()(1, ⟨7⟩) → 1:3
⟨34⟩ = iterate(⟨33⟩) → (1, 1)
⟨36⟩ = getfield(⟨34⟩, 1) → 1
⟨37⟩ = getfield(⟨34⟩, 2) → 1
⟨42⟩ = z[1] ~ DiscreteNonParametric(⟨31⟩) → 2
```

where `θ` stands for the environment of random variable assignments.
"""
abstract type Cont end


abstract type ArgSpec <: Cont end

struct Fixed{T} <: ArgSpec
    value::T
end
Base.show(io::IO, arg::Fixed) = print(io, arg.value)
(arg::Fixed)(θ) = arg.value

struct Variable{TV<:VarName} <: ArgSpec
    vn::TV
end
Base.show(io::IO, arg::Variable) = print(io, "θ[", arg.vn, "]")
(arg::Variable)(θ) = _lookup(θ, arg.vn)

function _lookup(θ, varname)
    if haskey(θ, varname)
        result = getindex(θ, varname)
        return result
    else
        # in the case of looking up x[i] with stored x,
        # simply do it the slow way and check all elements
        for (vn, value) in θ
            if DynamicPPL.subsumes(vn, varname)
                result = foldl((x, i) -> getindex(x, i...),
                               DynamicPPL.getindexing(varname),
                               init=value)
                return result
            end
        end
        throw(BoundsError(θ, varname))
    end
end


struct Transformation{TF, N, TArgs<:NTuple{N, Cont}} <: Cont
    f::TF
    args::TArgs
end

function Base.show(io::IO, t::Transformation)
    print(io, t.f, "(")
    join(io, t.args, ", ")
    print(io, ")")
end

(t::Transformation)(θ) = t.f((arg(θ) for arg in t.args)...)

function (t::Transformation{typeof(getindex)})(θ)
    # intercept this to simplify getindex(x, i) to direct lookup of x[i]
    array, indexing = first(t.args), Base.tail(t.args)
    if array isa Variable
        actual_vn = VarName(array.vn, (Tuple(ix(θ) for ix in indexing),))
        return _lookup(θ, actual_vn)
    else
        return t.f((arg(θ) for arg in t.args)...)
    end
end


struct LogLikelihood{TDist, TF, TArgs, TVal} <: Cont
    dist::TDist
    f::TF # function that was used to construct the distribution
    args::TArgs
    value::TVal
    
    function LogLikelihood(dist::Distribution, f, args::NTuple{N, Cont}, value) where {N}
        return new{typeof(dist), typeof(f), typeof(args), typeof(value)}(
            dist, f, args, value)
    end
end

function Base.show(io::IO, ℓ::LogLikelihood{D}) where {D}
    print(io, "logpdf(", ℓ.f, "(")
    join(io, ℓ.args, ", ")
    print(io, "), ", ℓ.value, ")")
end

(ℓ::LogLikelihood{D})(θ) where {D} = begin
    logpdf(ℓ.f((arg(θ) for arg in ℓ.args)...), ℓ.value(θ))
end


_init(::Tuple{}) = ()
_init(t::Tuple{Any}) = ()
_init(t::Tuple) = (first(t), _init(Base.tail(t))...)
_last(::Tuple{}) = ()
_last((x,)::Tuple{Any}) = x
_last(t::Tuple) = _last(Base.tail(t))



"""
    continuations(graph)

Assign to each node in the graph a `Cont` object, representing a function from a variable assignment
(θ) to the value of an expression or the log-likelihood in case of a tilde statement.
"""
function continuations(graph)
    c = SortedDict{Reference, Cont}()
    
    function convertarg(arg)
        if arg isa Reference
            cont = c[arg]
            stmt = graph[arg]
            if cont isa LogLikelihood && !isnothing(stmt.vn)
                return Variable(stmt.vn)
            elseif cont isa Transformation{typeof(getindex)} && !isnothing(stmt.definition)
                # two cases:
                # - vn[ix] = getindex(<a>, ix) where <a> = vn ~ D: then we keep getindex(θ[a], ix)
                #   (the whole array was sampled by the tilde)
                # - vn[ix] = getindex(<a>, ix) where <scalar> = vn[ix] ~ D: then we rewrite to
                # `getindex(θ[a], ix)`  (the array element was sampled individually)
                # this truncation is realized by `_init`, which discards the last element of a tuple,
                # and maps the empty tuple to itself.
                location, ref = stmt.definition
                array, ix = stmt.args[1], stmt.args[2:end]
                parent_vn = VarName(graph[ref].vn, _init(DynamicPPL.getindexing(graph[ref].vn)))
                parent_array = Variable(parent_vn)
                return Transformation(getindex, (parent_array, convertarg.(ix)...))
            else
                return cont
            end
        else
            return Fixed(arg)
        end
    end
    
    for (ref, stmt) in graph
        if stmt isa Union{Assumption, Observation}
            dist_stmt = stmt.dist
            dist = getvalue(dist_stmt)
            if dist_stmt isa Call
                f, args = dist_stmt.f, convertarg.(dist_stmt.args)
            elseif f isa Constant
                # fake a constant here... `getindex(Ref(x)) == x`
                f = getindex
                args = (Fixed(Ref(dist)),)
            end
            value = Variable(stmt.vn)
            c[ref] = LogLikelihood(dist, f, args, value)
        elseif stmt isa Call
            f, args = stmt.f, convertarg.(stmt.args)
            c[ref] = Transformation(f, args)
        elseif stmt isa Constant
            c[ref] = Fixed(getvalue(stmt))
        end
    end

    return c
end

"""
    conditionals(graph, varname)

Calculate the `GibbsConditional`s of all variables in `graph` that match `varname`.  There can 
be multiple observations for which this holds, so a dictionary is returned.
"""
function conditionals(graph, varname)
    dists = Dict{VarName, LogLikelihood}()
    blankets = DefaultDict{VarName, Vector{Pair{VarName, LogLikelihood}}}(
        Vector{Pair{Tuple, LogLikelihood}})
    conts = continuations(graph)
    
    for (ref, stmt) in graph
        if stmt isa Union{Assumption, Observation} && !isnothing(stmt.vn)
            # record distribution of this tilde if it matches the searched vn
            if DynamicPPL.subsumes(varname, stmt.vn)
                dists[stmt.vn] = conts[ref]
            end
            
            # add likelihood to all parents of which this RV is in the blanket
            for (pvn, p) in parent_variables(graph, stmt)
                for vn in keys(dists)
                    if DynamicPPL.subsumes(vn, p.vn)
                        push!(blankets[vn], pvn => conts[ref])
                        break
                    end
                end
            end
        end
    end

    return Dict(vn => GibbsConditional(vn, d, blankets[vn]) for (vn, d) in dists)
end


"""
    sampled_values(graph)

Extract the values of all observed and assumed random variables in the graph, including their
occuring parts.
"""
function sampled_values(graph)
    θ = Dict{VarName, Any}()
    for (ref, stmt) in graph
        if stmt isa Union{Assumption, Observation} && !isnothing(stmt.vn)
            θ[stmt.vn] = tovalue(graph, getvalue(stmt))
        elseif stmt isa Call && !isnothing(stmt.definition)
            # remember all intermediate RV values (including redundant `getindex` calls,
            # for simplicity)
            vn, _ = stmt.definition
            θ[vn] = getvalue(stmt)
        end
    end

    return θ
end


struct GibbsConditional{
    TVar<:VarName,
    TBase<:LogLikelihood,
    TBlanket}

    vn::TVar
    base::TBase
    blanket::TBlanket
end

function Base.show(io::IO, c::GibbsConditional)
    print(io, c.base)
    if !isempty(c.blanket)
        print(io, " + ")
        join(io, (β for (vn, β) in c.blanket), " + ")
    end
end


"""
    (c::GibbsConditional)(θ)

Return the conditional distribution of `vn` given the values fixed in `θ`, calculated by 
normalization over the conditional distribution of `vn` itlself and its Markov blanket.

Constructed as

    P[X = x | θ] ∝ P[X = x | parents(X)] * P[children(X) | parents(children(X))]

equivalent to a distribution `D` such that

    logpdf(D, x) = ℓ(x | θ) + ∑ blanketᵢ(x | θ)

where the factors `blanketᵢ` are the log probabilities in the Markov blanket.

This only works on discrete distributions, either scalar ones (resulting in a 
`DiscreteNonparametric`) or products of them (resulting in a `Product` of `DiscreteNonparametric`).
"""
function (c::GibbsConditional{V, L})(θ) where {
    V<:VarName, L<:LogLikelihood{<:DiscreteUnivariateDistribution}}

    Ω = try
        support(c.base.dist)
    catch
        throw(ArgumentError("Unable to get the support of $(c.base.dist) (probably infinite)!"))
    end

    θs_on_support = fixvalues(θ, c.vn => Ω)
    logtable = [c.base(θ′) + reduce(+, (β(θ′) for (vn, β) in c.blanket), init=0.0)
                for θ′ in θs_on_support]
    conditional = DiscreteNonParametric(Ω, softmax!(logtable))
    return conditional
end

# `Product`s can be treated as an array of iid variables
function (c::GibbsConditional{V, L})(θ) where {
    V<:VarName, L<:LogLikelihood{<:Product}}

    independent_distributions = c.base.dist.v
    Ωs = try
        support.(independent_distributions)
    catch
        throw(ArgumentError("Unable to get the support of $(c.base.dist) (probably infinite)!"))
    end
    
    conditionals = similar(Ωs, DiscreteNonParametric)
    
    for index in eachindex(Ωs, independent_distributions, conditionals)
        sub_vn = DynamicPPL.VarName(c.vn, (DynamicPPL.getindexing(c.vn)..., (index,)))
        θs_on_support = fixvalues(θ, sub_vn => Ωs[index])
        logtable = map(θs_on_support) do θ′
            c.base(θ′) + reduce(+, (β(θ′) for (vn, β) in c.blanket if vn == sub_vn), init=0.0)
        end
        conditionals[index] = DiscreteNonParametric(Ωs[index], softmax!(vec(logtable)))
    end

    return Product(conditionals)
end


# # Special treatment for CRP variables: calculate likelihoods as normal for truncated support
# # (covering all existing clusters), and marginalize the creation of a new cluster
# function (c::GibbsConditional{V, L})(θ) where {
#     V<:VarName, L<:LogLikelihood{<:ChineseRestaurantProcess}}
#     Ω = support(c.base.dist)
#     Ω_init, Ω_last = Ω[1:end-1], Ω[end]

#     θs_on_init = fixvalues(θ, c.vn => Ω_init)
#     logtable_init = Float64[c.base(θ′) + reduce(+, (β(θ′) for (vn, β) in c.blanket), init=0.0)
#                             for θ′ in θs_on_support]
    
#     θ_on_last = fixvalues(θ, c.vn => [Ω_last])
#     log_last = _estimate_last_likelihood(c, θ_on_last)
#     conditional = DiscreteNonParametric(Ω, softmax!(push!(logtable_init, log_last)))
#     return conditional
# end


# """
# Estimate the "new cluster" likelihood of a CRP, given through

#     𝓅(zₙ = K + 1 | z₁, ..., zₙ₋₁, μ, xₙ) ∝ (∏_{i = z ≥ n} 𝓅(zᵢ | z₁,...,zᵢ)) 𝓅(xₙ | zₙ = K + 1, μ),

# by approximating

#     𝓅(xₙ | zₙ = K + 1, μ) = ∫ 𝓅(xₙ | μ = m) dm ≈ 𝓅(xₙ | m)

# where Law(m) = Law(μ).
# """
# function _estimate_last_likelihood(c, θ)
#     l = c.base(θ)
    
#     for (ix, β) in c.blanket
#         if β isa LogLikelihood{<:ChineseRestaurantProcess} && β.dist.rpm == c.base.rpm
#             l += β(θ)
#         else
            
#             conditioned_dist = β.f((arg(θ) for arg in β.args)...)
#             sample = rand(conditioned_dist)
#             l += logpdf(conditioned_dist, sample)
#         end
#     end
    
#     return l
# end


(c::GibbsConditional)(θ) =
    throw(ArgumentError("Cannot condition a non-discrete or non-univariate distribution $(c.base)."))


"""Produce `|Ω|` copies of `θ` with the `fixedvn` entries fixed to the values in the support `Ω`."""
function fixvalues(θ, (source_vn, Ω))
    # "source" is the value stored in Ω; "target" is the matching value in θ.

    result = [copy(θ) for _ in Ω]
    
    for (θ′, value) in zip(result, Ω)
        for target_vn in keys(θ′)
            source_subsumes_target = DynamicPPL.subsumes(source_vn, target_vn)
            target_subsumes_source = DynamicPPL.subsumes(target_vn, source_vn)
            
            if source_subsumes_target && target_subsumes_source
                # target <- source, target[i] <- source[i]
                # both indices match -- we just update the complete thing
                θ′[target_vn] = value
                
            elseif target_subsumes_source
                # target <- source[i], target[i] <- source[i][j]
                # updating the target from a part of the source with "copy on write",
                # target[i][j] = source[i][j] <=> setindex!(copy(target[i]), source[i][j], j)

                # NB: we index into the target by the source index!
                target_indexing = DynamicPPL.getindexing(source_vn)
                ti_init, ti_last = _splitindexing(target_indexing)
                target = θ′[target_vn]
                initial_target = foldl((x, i) -> getindex(x, i...),
                                       ti_init,
                                       init=target)
                if isnothing(ti_last)
                    θ′[target_vn] = initial_target
                else
                    target = setindex!(copy(initial_target), value, ti_last...)
                    θ′[target_vn] = target
                end
            elseif source_subsumes_target
                # target[i][j] <- source[i]
                # setting the target to part of the source array
                # target[i][j] = source[i][j]
                
                target_indexing = DynamicPPL.getindexing(target_vn)
                target = foldl((x, i) -> getindex(x, i...),
                               target_indexing,
                               init=value)
                θ′[target_vn] = target
            end
        end
    end

    return result
end


_splitindexing(::NTuple{0}) = (), nothing
_splitindexing((i,)::NTuple{1}) = (), i
_splitindexing((i, j)::NTuple{2}) = (i,), j
_splitindexing(t::Tuple) = let (init, last) = _splitindexing(Base.tail(t))
    (first(t), init...), last
end


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

softmax(x::AbstractArray{<:AbstractFloat}) = softmax!(copy(x))



# D_w = Dirichlet(2, 1.0)
# w = rand(D_w)
# D_p = DiscreteNonParametric([0.3, 0.6], w)
# p = rand(D_p)
# D_x = Bernoulli(p)
# x = rand(D_x)
# conditioned(D_p, [logpdf(D_x, x)])
