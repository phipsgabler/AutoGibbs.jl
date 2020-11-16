# AutoGibbs

[![Build Status](https://travis-ci.com/phipsgabler/AutoGibbs.jl.svg?branch=master)](https://travis-ci.com/phipsgabler/AutoGibbs.jl)

Beware: this is nothing but a proof of concept.  Specifically, due to unfortate choices in
`IRTracker`, handling traces of models with data sets not particularly small can lead to exploding
type inference time, making the method unusable in some cases.

There are some limitations with respect to kinds of observations supported: 
- Only “simple” indexing is allowed
- Slices may work, everything else is at your own risk
- Broadcasted sampling is not working

## Dependency extraction in DPPL models

The first thing this library does is to slice out dependency graphs of the random variables in a
[DynamicPPL](https://github.com/TuringLang/DynamicPPL.jl) model:

```julia
@model function test0(x)
    λ ~ Gamma(2.0, inv(3.0))
    m ~ Normal(0, sqrt(1 / λ))
    x ~ Normal(m, sqrt(1 / λ))
end

graph0 = trackdependencies(test0(1.4))
```

will result in 

```
⟨2⟩ = 1.4
⟨3⟩ = Gamma(2.0, 0.3333333333333333) → Gamma{Float64}(α=2.0, θ=0.3333333333333333)
⟨4⟩ = λ ~ ⟨3⟩ → 0.9881137307157533
⟨5⟩ = /(1, ⟨4⟩) → 1.0120292522153667
⟨6⟩ = sqrt(⟨5⟩) → 1.0059966462247112
⟨7⟩ = Normal(0, ⟨6⟩) → Normal{Float64}(μ=0.0, σ=1.0059966462247112)
⟨8⟩ = m ~ ⟨7⟩ → 1.0814646155873082
⟨9⟩ = /(1, ⟨4⟩) → 1.0120292522153667
⟨10⟩ = sqrt(⟨9⟩) → 1.0059966462247112
⟨11⟩ = Normal(⟨8⟩, ⟨10⟩) → Normal{Float64}(μ=1.0814646155873082, σ=1.0059966462247112)
⟨12⟩ = x ⩪ ⟨11⟩ ← ⟨2⟩
```

Notation: references to intermediate values are written in ⟨angle brackets⟩.  The names of random
variables in tilde statements are annotated with names (and possibly indices).  Unobserved random
variables (“assumptions”) are notated by a simple `~`, observed ones use `⩪`.  Equality signs denote
assignment of intermediate deterministic values, as in normal SSA-form IR.  The results of
deterministic or assumed values come after the `→`; `←` is used for values that are observed.

Numbers of references are quite contiguously numbered, but some numbers may be missing, since they
get removed by the slicing process.

The result of `trackdependencies` is a `Graph`, which is essentially an ordered dictionary from
`Reference`s to statements, that can be either of `Assumption`, `Observation`, `Call`, or
`Constant`.  For debugging, you can also use integers to index the graph:

```
julia> graph0[4]
λ ~ ⟨3⟩ → 0.9881137307157533
```

You can also write the graph to a GraphViz dot file:

```
julia> AutoGibbs.savedot("/tmp/graph.dot", graph0)
```

and covert it to PDF with

```
# Noto fonts are only required for fancy unicode, but otherwise some things will look ugly.
dot /tmp/graph.dot -Tpdf -Nfontname="Noto Mono" -Efontname="Noto Mono" > /tmp/graph.pdf
```

![Dependency graph output](./images/graph.png)

