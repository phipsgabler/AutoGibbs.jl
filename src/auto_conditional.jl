using AbstractMCMC
using DynamicPPL
using Random
using Turing


export AutoConditional


"""
    AutoConditional(sym)

A "pseudo-sampler" to use automatically extracted conditionals within `Gibbs`.  The conditionals are
extracted from the model trace at sampling step and allow dynamically changing models.  If you don't
need that, use `StaticConditional` instead.

`AutoConditional(:x)` will sample the Gibbs conditional of variable `x`, or equivalently, all 
variables subsumed by `x`.

# Examples

```julia

@model function test(x)
    w ~ Dirichlet(2, 1.0)
    p ~ DiscreteNonParametric([0.3, 0.7], w)
    x ~ Bernoulli(p)
end

m = test(false)

sample(m, Gibbs(AutoConditional(:p), MH(:w)), 10)
```
"""
struct AutoConditional{S}
    AutoConditional(sym::Symbol) = new{sym}()
end

DynamicPPL.getspace(::AutoConditional{S}) where {S} = (S,)
DynamicPPL.alg_str(::AutoConditional) = "AutoConditional"
Turing.isgibbscomponent(::AutoConditional) = true


function Turing.Sampler(
    alg::AutoConditional,
    model::Turing.Model,
    s::DynamicPPL.Selector=DynamicPPL.Selector()
)
    state = Turing.Inference.SamplerState(Turing.VarInfo(model))
    return Turing.Sampler(alg, Dict{Symbol, Any}(), s, state)
end


function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Turing.Model,
    spl::Turing.Sampler{<:AutoConditional{S}},
    N::Integer,
    transition;
    kwargs...
) where {S}
    if spl.selector.rerun # Recompute joint in logp
        model(spl.state.vi)
    end
    
    graph = trackdependencies(model, spl.state.vi)
    conditioned_vn = DynamicPPL.VarName(S)
    values = sampled_values(graph)
    for (vn, cond) in conditionals(graph, conditioned_vn)
        updated = rand(rng, cond(values))
        spl.state.vi[vn] = [updated;]
    end
    
    return transition
end
