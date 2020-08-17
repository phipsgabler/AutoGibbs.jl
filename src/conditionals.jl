using DataStructures: DefaultDict
using Distributions
using DynamicPPL


export conditional_dists


"""
    abstract type Cont end

`Cont` is an analytic representation of joint distribution represented in a `Graph` -- a function
from a dictionary of variable assignments to a log probability. 

 Each `Call` gets associated with a `Transformation`, and each tilde statement with a `LogLikelihood`.  
Those are both callable with a dict argument.  SSA variables get transformed to either a `Fixed` value 
for constants, a `Variable` for assignments of random variables, or the `Cont` they come from.

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
(arg::Variable)(θ) = getindex(θ, arg.vn)


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


struct LogLikelihood{TDist<:Distribution, TVal, TArgs<:Tuple} <: Cont
    dist::TDist
    value::TVal
    args::TArgs
    
    function LogLikelihood(dist::D, value, args::NTuple{N, Cont}) where {D<:Distribution, N}
        return new{D, typeof(value), typeof(args)}(dist,value, args)
    end
end

function Base.show(io::IO, ℓ::LogLikelihood{D}) where {D}
    print(io, "logpdf(", _shortname(D), "(")
    join(io, ℓ.args, ", ")
    print(io, "), ", ℓ.value, ")")
end

(ℓ::LogLikelihood{D})(θ) where {D} = logpdf(D((arg(θ) for arg in ℓ.args)...), ℓ.value(θ))


function continuations(graph)
    c = SortedDict{Reference, Cont}()
    
    function convertarg(arg)
        if arg isa Reference
            cont = c[arg]
            stmt = graph[arg]
            if cont isa LogLikelihood && !isnothing(stmt.vn)
                Variable(stmt.vn)
            elseif cont isa Transformation && !isnothing(stmt.definition)
                Variable(stmt.definition[1])
            else
                cont
            end
        else
            Fixed(arg)
        end
    end
    
    for (ref, stmt) in graph
        if stmt isa Union{Assumption, Observation}
            dist_call = stmt.dist
            dist = getvalue(dist_call)

            if dist_call isa Call
                args = convertarg.(dist_call.args)
            elseif dist_call isa Constant
                args = Fixed.(params(dist))
            end

            if stmt isa Observation
                value = convertarg(getvalue(stmt))
            else
                value = Variable(stmt.vn)
            end
            
            c[ref] = LogLikelihood(dist, value, args)
            
        # elseif stmt isa Call{<:Tuple, typeof(getindex)}
        #     vn, compound_ref = stmt.definition
        #     ix = getindexing(vn)[1]
        #     f, args = stmt.f, convertarg.(stmt.args)
        #     dist = graph[compound_ref].v[ix...]
        #     value = Variable(VarName(graph[compound_ref].vn, (ix,)))
        #     c[ref] = LogLikelihood(typeof(dist), value, )
        elseif stmt isa Call
            f, args = stmt.f, convertarg.(stmt.args)
            c[ref] = Transformation(f, args)
        elseif stmt isa Constant
            c[ref] = Fixed(getvalue(stmt))
        end
    end

    return c
end


function conditionals(graph, varname)
    # There can be multiple tildes for one `varname`, e.g., `x[1], x[2]` both subsumed by `x`.
    # dists = Dict{VarName, Distribution}()
    # blankets = DefaultDict{Tuple{VarName, Union{Nothing, Tuple}}, Float64}(0.0)
    dists = Dict{VarName, LogLikelihood}()
    blankets = DefaultDict{VarName, Vector{Pair{Tuple, LogLikelihood}}}(
        Vector{Pair{Tuple, LogLikelihood}})
    θ = Dict{VarName, Any}()
    conts = continuations(graph)
    
    for (ref, stmt) in graph
        if stmt isa Union{Assumption, Observation} && !isnothing(stmt.vn)
            θ[stmt.vn] = getvalue(stmt)
            
            # record distribution of this tilde if it matches the searched vn
            if DynamicPPL.subsumes(varname, stmt.vn)
                dists[stmt.vn] = conts[ref]
            end
            
            # add likelihood to all parents of which this RV is in the blanket
            for (p, ix) in parent_variables(graph, stmt)
                for vn in keys(dists)
                    if DynamicPPL.subsumes(vn, p.vn)
                        push!(blankets[vn], ix => conts[ref])
                        break
                    end
                end
            end
            
        elseif stmt isa Call && !isnothing(stmt.definition)
            # remember all intermediate RV values (including redundant `getindex` calls,
            # for simplicity)
            vn, _ = stmt.definition
            θ[vn] = getvalue(stmt)
        end
    end

    return Dict{VarName, Distribution}(
        vn => GibbsConditional(vn, d, blankets[vn], θ) for (vn, d) in dists)
end


struct GibbsConditional{
    F<:VariateForm,
    S<:ValueSupport,
    TCond<:Distribution{F, S},
    TBase,
    TBlanket,
    TValues} <: Distribution{F, S}
    
    conditional::TCond
    base::TBase
    blanket::TBlanket
    θ::TValues
end

function Base.show(io::IO, c::GibbsConditional)
    print(io, c.base)
    if !isempty(c.blanket)
        print(io, " + ")
        join(io, (β for (ix, β) in c.blanket), " + ")
    end
end

GibbsConditional(vn, ℓ, blanket, θ) =
    GibbsConditional(conditioned(vn, ℓ, blanket, θ), ℓ, blanket, θ)

Distributions.rand(rng::AbstractRNG, c::GibbsConditional) = rand(rng, c.conditional)
Distributions.logpdf(c::GibbsConditional, x) = logpdf(c.conditional, x)

"""
    conditioned(vn, ℓ, blanket, θ)

Return the conditional distribution of `vn` given the values fixed in `θ`, calculated by 
normalization over the likelihood `ℓ` and Markov blanket likelihoods `blanket`.

Constructed as

    P[X = x | θ] ∝ P[X = x | parents(X)] * P[children(X) | parents(children(X))]

equivalent to a distribution `D` such that

    logpdf(D, x) = ℓ(x | θ) + ∑ blanketᵢ(x | θ)

where the factors `blanketᵢ` are the log probabilities in the Markov blanket.

This only works on discrete distributions, either scalar ones (resulting in a 
`DiscreteNonparametric`) or products of them (resulting in a `Product` of `DiscreteNonparametric`).
"""
function conditioned(vn::VarName, ℓ::LogLikelihood{<:DiscreteUnivariateDistribution}, blanket, θ)
    local Ω

    try
        Ω = support(ℓ.dist)
    catch
        throw(ArgumentError("Unable to get the support of $(ℓ.dist) (probably infinite)!"))
    end

    θs_on_support = fixvalues(vn, θ, Ω)
    ℓ_base = ℓ(θ)
    logtable = [ℓ_base + reduce(+, (β(θ) for (ix, β) in blanket), init=zero(ℓ_base)) for θ in θs_on_support]
    # @show logpdf.(d0, Ω)
    # @show (softmax(logtable))
    conditional = DiscreteNonParametric(Ω, softmax!(logtable))
    return conditional
end

# `Product`s can be treated as an array of iid variables
# function conditioned(vn::VarName, ℓ::LogLikelihood{<:Product}, blanket, θ)
    # return Product([conditioned]conditioned.(ℓ.dist.v, blanket, θ))
# end

conditioned(vn::VarName, ℓ::LogLikelihood, blanket, θ) =
    throw(ArgumentError("Cannot condition a non-discrete or non-univariate distribution $(ℓ.dist)."))


"""Produce `|Ω|` copies of `θ` with the `vn` entries fixed to the values in the support `Ω`."""
function fixvalues(vn, θ, Ω)
    # foldl((x, i) -> getindex(x, i...),
          # DynamicPPL.getindexing(vn),
    # init=x)
    result = [copy(θ) for ω in Ω]
    for i in eachindex(result)
        θ′ = result[i]
        for variable in keys(θ′)
            if DynamicPPL.subsumes(vn, variable)
                updated = copy(θ′[variable])
                
                θ′[variable] = updated
            end
        end
    end

    return result
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
