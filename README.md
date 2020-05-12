# AutoGibbs

[![Build Status](https://travis-ci.com/phipsgabler/AutoGibbs.jl.svg?branch=master)](https://travis-ci.com/phipsgabler/AutoGibbs.jl)


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
⟨4:λ⟩ ~ Gamma(2.0, 0.3333333333333333) → 1.0351608689025245
⟨5⟩ = /(1, ⟨4:λ⟩) → 0.9660334253749353
⟨6⟩ = sqrt(⟨5⟩) → 0.982869994137035
⟨8:m⟩ ~ Normal(0, ⟨6⟩) → -2.0155543806491205
⟨9⟩ = /(1, ⟨4:λ⟩) → 0.9660334253749353
⟨10⟩ = sqrt(⟨9⟩) → 0.982869994137035
⟨12:x⟩ ⩪ Normal(⟨8:m⟩, ⟨10⟩) ← ⟨2⟩
```

Notation: references to values are written in ⟨angle brackets⟩, with names (and possibly indices) or
random random variables after a colon.  Unobserved random variables (“assumptions”) are notated by a
simple `~`, observed ones use `⩪`.  Equality signs denote assignment of intermediate deterministic
values.  The results of deterministic or assumed values come after the `→`; `←` is used for values
that are observed.

(Numbers of references are quite contiguously numbered, but some numbers may be missing, since they
get removed by the slicing process.)

The result of `trackdependencies` is a `Graph`, which is essentially an ordered dictionary from
`Reference`s to statements, that can be either of `Assumption`, `Observation`, `Call`, or
`Constant`.  You can also use just itegers to index the graph:

```
julia> graph0[4]
Gamma(2.0, 0.3333333333333333) → 1.0351608689025245
```

You can also write the graph to a GraphViz dot file:

```
julia> AutoGibbs.savedot("/tmp/graph.dot", graph0)

shell> dot /tmp/graph.dot -Tpdf > /tmp/graph.pdf
```

But don’t expect any magic to make it look good:

![Dependency graph output](./images/graph.png)

