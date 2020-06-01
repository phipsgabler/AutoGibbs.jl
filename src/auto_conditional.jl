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
    AutoConditional(sym::Symbol) = new{sym}(conditional)
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
    graph = trackdependencies(model, spl.state.vi)
    vn = DynamicPPL.VarName(S)
    conddists = conditional_dists(graph, vn)
    updated = rand.(rng, conddist)  
    spl.state.vi[vn] = updated
    
    return transition
end
