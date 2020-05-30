using AbstractMCMC
using DynamicPPL
using Turing


"""
    AutoConditional(sym)

A "pseudo-sampler" to use automatically extracted conditionals within `Gibbs`.
`AutoConditional(:x)` will sample the Gibbs conditional of variable `x`.

# Examples

```julia

@model test(x) = begin
    w ~ Dirichlet(2, 1.0)
    p ~ DiscreteNonParametric([0.3, 0.7], w)
    x ~ Bernoulli(p)
end

m = test(false)

sample(m, Gibbs(AutoConditional(:p), MH(:w)), 10)
```
"""
struct AutoConditional{S}
    function AutoConditional(sym::Symbol)
        return new{sym}(conditional)
    end
end

DynamicPPL.getspace(::AutoConditional{S}) where {S} = (S,)
Turing.alg_str(::AutoConditional) = "AutoConditional"
Turing.isgibbscomponent(::AutoConditional) = true


function Turing.Sampler(
    alg::AutoConditional,
    model::Turing.Model,
    s::Turing.Selector=Turing.Selector()
)
    return Turing.Sampler(alg, Dict{Symbol, Any}(), s, Turing.SamplerState(Turing.VarInfo(model)))
end


function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Turing.Model,
    spl::Turing.Sampler{<:AutoConditional{S}},
    N::Integer,
    transition;
    kwargs...
) where {S}
    deps = trackdependencies(model, spl.state.vi)
    conddist = conditional_dist(deps, S)
    updated = rand(rng, conddist)
    spl.state.vi[DynamicPPL.VarName(S)] = [updated;]  # setindex allows only vectors in this case...
    
    return transition
end


"""
    conditioned(θ::NamedTuple, ::Val{S})

Extract a `NamedTuple` of the values in `θ` conditioned on `S`; i.e., all names of `θ` except for
`S`, mapping to their respecitve values.

`θ` is assumed to come from `tonamedtuple(vi)`, which returns a `NamedTuple` of the form

```julia
t = (m = ([0.234, -1.23], ["m[1]", "m[2]"]), λ = ([1.233], ["λ"])
```

so this function does both the cleanup of indexing and filtering by name. `conditioned(t, Val{m}())`
and `conditioned(t, Val{λ}())` will therefore return

```julia
(λ = 1.233,)
```

and

```julia
(m = [0.234, -1.23],)
```
"""
@generated function conditioned(θ::NamedTuple{names}, ::Val{S}) where {names, S}
    condvals = [:($n = extractparam(θ.$n)) for n in names if n ≠ S]
    return Expr(:tuple, condvals...)
end


"""Takes care of removing the `tonamedtuple` indexing form."""
extractparam(p::Tuple{Vector{<:Array{<:Real}}, Vector{String}}) = foldl(vcat, p[1])
function extractparam(p::Tuple{Vector{<:Real}, Vector{String}})
    values, strings = p
    if length(values) == length(strings) == 1 && !occursin(r".\[.+\]$", strings[1])
        # if m ~ MVNormal(1, 1), we could have have ([1], ["m[1]"])!
        return values[1]
    else
        return values
    end
end
