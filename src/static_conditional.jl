using AbstractMCMC
using DynamicPPL
using Random
using Turing

export StaticConditional


"""
    StaticConditional(sym)

A "pseudo-sampler" to use automatically extracted conditionals within `Gibbs`.  This works by 
calculating the conditionals once on the model and reusing them in every sampling step; hence, the
trace of the model must keep the same at every step.

`StaticConditional(:x)` will sample the Gibbs conditional of variable `x`, or equivalently, all 
variables subsumed by `x`.

# Examples

```julia

@model test(x) = begin
    w ~ Dirichlet(2, 1.0)
    p ~ DiscreteNonParametric([0.3, 0.7], w)
    x ~ Bernoulli(p)
end

m = test(false)
cond_p = StaticConditional(m, :p)
sample(m, Gibbs(cond_p, MH(:w)), 10)
```
"""
struct StaticConditional{S, TCond}
    conditionals::TCond
    
    function StaticConditional(model::Turing.Model, sym::Symbol)
        vn = VarName(sym)
        graph = trackdependencies(model)
        conds = conditionals(graph, vn)
        return new{sym, typeof(conds)}(conds)
    end
end

DynamicPPL.getspace(::StaticConditional{S}) where {S} = (S,)
DynamicPPL.alg_str(::StaticConditional) = "StaticConditional"
Turing.isgibbscomponent(::StaticConditional) = true


function Turing.Sampler(
    alg::StaticConditional,
    model::Turing.Model,
    s::DynamicPPL.Selector=DynamicPPL.Selector()
)
    state = Turing.Inference.SamplerState(Turing.VarInfo(model))
    return Turing.Sampler(alg, Dict{Symbol, Any}(), s, state)
end


function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Turing.Model,
    spl::Turing.Sampler{<:StaticConditional},
    N::Integer,
    transition;
    kwargs...
)
    if spl.selector.rerun # Recompute joint in logp
        model(spl.state.vi)
    end
    
    vi = spl.state.vi
    values = sampled_values(model, vi)
    for (vn, cond) in spl.alg.conditionals
        updated = rand(rng, cond(values))
        vi[vn] = [updated;]
    end
    
    
    return transition
end


function sampled_values(model::Turing.Model, vi::DynamicPPL.UntypedVarInfo)
    assumed_vars = Dict(vn => vi[vn] for vn in keys(vi))
    observed_vars = Dict(VarName(arg) => value for (arg, value) in pairs(model.args))
    return merge(assumed_vars, observed_vars)
end

function sampled_values(model::Turing.Model, vi::DynamicPPL.TypedVarInfo)
    assumed_vars = Dict(vn => vi[vn]
                        for name_meta in values(vi.metadata)
                        for vn in name_meta.vns)
    observed_vars = Dict(VarName(arg) => value for (arg, value) in pairs(model.args))
    return merge(assumed_vars, observed_vars)
end

