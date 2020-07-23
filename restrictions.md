# Definitions on index expressions

“Simple” means: an `Integer` or a `UnitRange` of `Integer`s.

Restrictions on indices:

- Only simple indices: `x[i]`, `x[i:j]`, `x[y[i]]` where `y[i]` is simple are allowed, 
  `x[[i, j]]` and `x[y[i]]` where `y[i]` is not simple are not.
- No nested indexing: `x[i,j]` for simple `i, j` are allowed, `x[i][j]` is not.


## Bernoulli mixture

```julia
@model function bernoulli_mixture(x)
    w ~ Dirichlet(2, 1.0)
    p ~ DiscreteNonParametric([0.3, 0.7], w)
    x ~ Bernoulli(p)
end
```

```
⟨2⟩ = false
⟨4:w⟩ ~ Dirichlet(2, 1.0) → [0.34504152757607964, 0.6549584724239205]
⟨6:p⟩ ~ DiscreteNonParametric([0.3, 0.7], ⟨4:w⟩) → 0.7
⟨8:x⟩ ⩪ Bernoulli(⟨6:p⟩) ← ⟨2⟩
```


## GMM

```julia
@model function gmm(x, K)
    N = length(x)

    μ ~ filldist(Normal(), K)
    w ~ Dirichlet(K, 1.0)
    z ~ filldist(Categorical(w), N)

    for n in eachindex(x)
        x[n] ~ Normal(μ[z[n]], 1.0)
    end
end
```

```
⟨2⟩ = [0.1, -0.05, 1.0]
⟨4⟩ = 2
⟨5⟩ = length(⟨2⟩) → 3
⟨7:μ⟩ ~ filldist(Normal{Float64}(μ=0.0, σ=1.0), ⟨4⟩) → [0.7228539921978293, 0.8513272040438701]
⟨9:w⟩ ~ Dirichlet(⟨4⟩, 1.0) → [0.5384636379324796, 0.46153636206752024]
⟨10⟩ = DiscreteNonParametric{Int64,P,Base.OneTo{Int64},Ps} where Ps where P(⟨9:w⟩) → DiscreteNonParametric{Int64,Float64,Base.OneTo{Int64},Array{Float64,1}}(support=Base.OneTo(2), p=[0.5384636379324796, 0.46153636206752024])
⟨12:z⟩ ~ filldist(⟨10⟩, ⟨5⟩) → [2, 1, 1]
⟨13⟩ = eachindex(⟨2⟩) → Base.OneTo(3)
⟨14⟩ = iterate(⟨13⟩) → (1, 1)
⟨16⟩ = getfield(⟨14⟩, 1) → 1
⟨17⟩ = getfield(⟨14⟩, 2) → 1
⟨24⟩ = getindex(⟨2⟩, ⟨16⟩) → 0.1
⟨25:x[⟨16⟩]⟩ ⩪ Normal(⟨7:μ[⟨12:z[⟨16⟩]⟩]⟩, 1.0) ← ⟨24⟩
⟨26⟩ = iterate(⟨13⟩, ⟨17⟩) → (2, 2)
⟨28⟩ = getfield(⟨26⟩, 1) → 2
⟨29⟩ = getfield(⟨26⟩, 2) → 2
⟨36⟩ = getindex(⟨2⟩, ⟨28⟩) → -0.05
⟨37:x[⟨28⟩]⟩ ⩪ Normal(⟨7:μ[⟨12:z[⟨28⟩]⟩]⟩, 1.0) ← ⟨36⟩
⟨38⟩ = iterate(⟨13⟩, ⟨29⟩) → (3, 3)
⟨40⟩ = getfield(⟨38⟩, 1) → 3
⟨47⟩ = getindex(⟨2⟩, ⟨40⟩) → 1.0
⟨48:x[⟨40⟩]⟩ ⩪ Normal(⟨7:μ[⟨12:z[⟨40⟩]⟩]⟩, 1.0) ← ⟨47⟩
```



## Hidden Markov Model

```julia
@model function hmm(x, K, ::Type{T}=Float64) where {T<:Real}
    # Get observation length.
    N = length(x)

    # State sequence.
    s = zeros(Int, N)

    # Emission matrix.
    m = Vector{T}(undef, K)

    # Transition matrix.
    T = Vector{Vector{T}}(undef, K)

    # Assign distributions to each element
    # of the transition matrix and the
    # emission matrix.
    for i = 1:K
        T[i] ~ Dirichlet(K, 1.0)
        m[i] ~ Normal(i, 0.5)
    end
    
    # Observe each point of the input.
    s[1] ~ Categorical(K)
    x[1] ~ Normal(m[s[1]], 0.1)

    for i = 2:N
        s[i] ~ Categorical(T[s[i-1]])
        x[i] ~ Normal(m[s[i]], 0.1)
    end
end
```

```
⟨2⟩ = [0.1, -0.05, 1.0]
⟨4⟩ = 2
⟨6⟩ = Float64
⟨7⟩ = length(⟨2⟩) → 3
⟨8⟩ = zeros(Int64, ⟨7⟩) → [0, 0, 0]
⟨9⟩ = apply_type(Array{T,1} where T, ⟨6⟩) → Array{Float64,1}
⟨10⟩ = ⟨9⟩(array initializer with undefined values, ⟨4⟩) → [6.95005462400823e-310, 5.0e-324]
⟨11⟩ = apply_type(Array{T,1} where T, ⟨6⟩) → Array{Float64,1}
⟨12⟩ = apply_type(Array{T,1} where T, ⟨11⟩) → Array{Array{Float64,1},1}
⟨13⟩ = ⟨12⟩(array initializer with undefined values, ⟨4⟩) → Array{Float64,1}[#undef, #undef]
⟨14⟩ = Colon()(1, ⟨4⟩) → 1:2
⟨15⟩ = iterate(⟨14⟩) → (1, 1)
⟨17⟩ = getfield(⟨15⟩, 1) → 1
⟨18⟩ = getfield(⟨15⟩, 2) → 1
⟨23:T[⟨17⟩]⟩ ~ Dirichlet(⟨4⟩, 1.0) → [0.2949890529242057, 0.7050109470757943]
⟨28:m[⟨17⟩]⟩ ~ Normal(⟨17⟩, 0.5) → 1.075678499757859
⟨29⟩ = iterate(⟨14⟩, ⟨18⟩) → (2, 2)
⟨31⟩ = getfield(⟨29⟩, 1) → 2
⟨36:T[⟨31⟩]⟩ ~ Dirichlet(⟨4⟩, 1.0) → [0.5265746011731416, 0.4734253988268585]
⟨41:m[⟨31⟩]⟩ ~ Normal(⟨31⟩, 0.5) → 2.326083830434567
⟨44:s[1]⟩ ~ DiscreteNonParametric(⟨4⟩) → 1
⟨49⟩ = getindex(⟨2⟩, 1) → 0.1
⟨50:x[1]⟩ ⩪ Normal(⟨10:m[⟨8:s[1]⟩]⟩, 0.1) ← ⟨49⟩
⟨51⟩ = Colon()(2, ⟨7⟩) → 2:3
⟨52⟩ = iterate(⟨51⟩) → (2, 2)
⟨54⟩ = getfield(⟨52⟩, 1) → 2
⟨55⟩ = getfield(⟨52⟩, 2) → 2
⟨56⟩ = -(⟨54⟩, 1) → 1
⟨63:s[⟨54⟩]⟩ ~ DiscreteNonParametric(⟨13:T[⟨8:s[⟨56⟩]⟩]⟩) → 2
⟨70⟩ = getindex(⟨2⟩, ⟨54⟩) → -0.05
⟨71:x[⟨54⟩]⟩ ⩪ Normal(⟨10:m[⟨8:s[⟨54⟩]⟩]⟩, 0.1) ← ⟨70⟩
⟨72⟩ = iterate(⟨51⟩, ⟨55⟩) → (3, 3)
⟨74⟩ = getfield(⟨72⟩, 1) → 3
⟨75⟩ = -(⟨74⟩, 1) → 2
⟨82:s[⟨74⟩]⟩ ~ DiscreteNonParametric(⟨13:T[⟨8:s[⟨75⟩]⟩]⟩) → 1
⟨89⟩ = getindex(⟨2⟩, ⟨74⟩) → 1.0
⟨90:x[⟨74⟩]⟩ ⩪ Normal(⟨10:m[⟨8:s[⟨74⟩]⟩]⟩, 0.1) ← ⟨89⟩
```



## Infinite Mixture Model


```julia
@model function imm(x)
    N = length(x)

    nk = zeros(Int, N)
    G = ChineseRestaurantProcess(DirichletProcess(1.0), nk)
    
    z = zeros(Int, length(x))
    for i in 1:N
        z[i] ~ G
        nk[z[i]] += 1
    end

    K = findlast(!iszero, nk)
    μ ~ filldist(Normal(), K)

    for n in 1:N
        x[n] ~ Normal(μ[z[n]], 1.0)
    end
end
```

```
⟨2⟩ = [0.1, -0.05, 1.0]
⟨3⟩ = length(⟨2⟩) → 3
⟨4⟩ = zeros(Int64, ⟨3⟩) → [0, 0, 0]
⟨6⟩ = length(⟨2⟩) → 3
⟨7⟩ = zeros(Int64, ⟨6⟩) → [0, 0, 0]
⟨8⟩ = Colon()(1, ⟨3⟩) → 1:3
⟨9⟩ = iterate(⟨8⟩) → (1, 1)
⟨11⟩ = getfield(⟨9⟩, 1) → 1
⟨12⟩ = getfield(⟨9⟩, 2) → 1
⟨16:z[⟨11⟩]⟩ ~ ChineseRestaurantProcess(DirichletProcess{Float64}(1.0), ⟨4⟩) → 1
⟨18⟩ = getindex(⟨4⟩, ⟨7:z[⟨11⟩]⟩) → 0
⟨19⟩ = +(⟨18⟩, 1) → 1
⟨20⟩ = setindex!(⟨4⟩, ⟨19⟩, ⟨7:z[⟨11⟩]⟩) → [1, 0, 0]
⟨21⟩ = iterate(⟨8⟩, ⟨12⟩) → (2, 2)
⟨23⟩ = getfield(⟨21⟩, 1) → 2
⟨24⟩ = getfield(⟨21⟩, 2) → 2
⟨28:z[⟨23⟩]⟩ ~ ChineseRestaurantProcess(DirichletProcess{Float64}(1.0), ⟨4⟩) → 1
⟨30⟩ = getindex(⟨4⟩, ⟨7:z[⟨23⟩]⟩) → 1
⟨31⟩ = +(⟨30⟩, 1) → 2
⟨32⟩ = setindex!(⟨4⟩, ⟨31⟩, ⟨7:z[⟨23⟩]⟩) → [2, 0, 0]
⟨33⟩ = iterate(⟨8⟩, ⟨24⟩) → (3, 3)
⟨35⟩ = getfield(⟨33⟩, 1) → 3
⟨39:z[⟨35⟩]⟩ ~ ChineseRestaurantProcess(DirichletProcess{Float64}(1.0), ⟨4⟩) → 1
⟨41⟩ = getindex(⟨4⟩, ⟨7:z[⟨35⟩]⟩) → 2
⟨42⟩ = +(⟨41⟩, 1) → 3
⟨43⟩ = setindex!(⟨4⟩, ⟨42⟩, ⟨7:z[⟨35⟩]⟩) → [3, 0, 0]
⟨44⟩ = findlast(#58, ⟨4⟩) → 1
⟨46:μ⟩ ~ filldist(Normal{Float64}(μ=0.0, σ=1.0), ⟨44⟩) → [-0.3781818869614393]
⟨47⟩ = Colon()(1, ⟨3⟩) → 1:3
⟨48⟩ = iterate(⟨47⟩) → (1, 1)
⟨50⟩ = getfield(⟨48⟩, 1) → 1
⟨51⟩ = getfield(⟨48⟩, 2) → 1
⟨58⟩ = getindex(⟨2⟩, ⟨50⟩) → 0.1
⟨59:x[⟨50⟩]⟩ ⩪ Normal(⟨46:μ[⟨7:z[⟨50⟩]⟩]⟩, 1.0) ← ⟨58⟩
⟨60⟩ = iterate(⟨47⟩, ⟨51⟩) → (2, 2)
⟨62⟩ = getfield(⟨60⟩, 1) → 2
⟨63⟩ = getfield(⟨60⟩, 2) → 2
⟨70⟩ = getindex(⟨2⟩, ⟨62⟩) → -0.05
⟨71:x[⟨62⟩]⟩ ⩪ Normal(⟨46:μ[⟨7:z[⟨62⟩]⟩]⟩, 1.0) ← ⟨70⟩
⟨72⟩ = iterate(⟨47⟩, ⟨63⟩) → (3, 3)
⟨74⟩ = getfield(⟨72⟩, 1) → 3
⟨81⟩ = getindex(⟨2⟩, ⟨74⟩) → 1.0
⟨82:x[⟨74⟩]⟩ ⩪ Normal(⟨46:μ[⟨7:z[⟨74⟩]⟩]⟩, 1.0) ← ⟨81⟩
```


## Changepoint model

```julia
@model function changepoint(y)
    α = 1/mean(y)
    λ1 ~ Exponential(α)
    λ2 ~ Exponential(α)
    τ ~ DiscreteUniform(1, length(y))
    for idx in 1:length(y)
        y[idx] ~ Poisson(τ > idx ? λ1 : λ2)
    end
end
```

```
⟨2⟩ = [1.1, 0.9, 0.2]
⟨3⟩ = mean(⟨2⟩) → 0.7333333333333334
⟨4⟩ = /(1, ⟨3⟩) → 1.3636363636363635
⟨6:λ1⟩ ~ Exponential(⟨4⟩) → 0.0279149494437247
⟨8:λ2⟩ ~ Exponential(⟨4⟩) → 1.0554317079928577
⟨9⟩ = length(⟨2⟩) → 3
⟨11:τ⟩ ~ DiscreteUniform(1, ⟨9⟩) → 3
⟨12⟩ = length(⟨2⟩) → 3
⟨13⟩ = Colon()(1, ⟨12⟩) → 1:3
⟨14⟩ = iterate(⟨13⟩) → (1, 1)
⟨16⟩ = getfield(⟨14⟩, 1) → 1
⟨17⟩ = getfield(⟨14⟩, 2) → 1
⟨23⟩ = getindex(⟨2⟩, ⟨16⟩) → 1.1
⟨24:y[⟨16⟩]⟩ ⩪ Poisson(⟨6:λ1⟩) ← ⟨23⟩
⟨25⟩ = iterate(⟨13⟩, ⟨17⟩) → (2, 2)
⟨27⟩ = getfield(⟨25⟩, 1) → 2
⟨28⟩ = getfield(⟨25⟩, 2) → 2
⟨34⟩ = getindex(⟨2⟩, ⟨27⟩) → 0.9
⟨35:y[⟨27⟩]⟩ ⩪ Poisson(⟨6:λ1⟩) ← ⟨34⟩
⟨36⟩ = iterate(⟨13⟩, ⟨28⟩) → (3, 3)
⟨38⟩ = getfield(⟨36⟩, 1) → 3
⟨44⟩ = getindex(⟨2⟩, ⟨38⟩) → 0.2
⟨45:y[⟨38⟩]⟩ ⩪ Poisson(⟨8:λ2⟩) ← ⟨44⟩
```

