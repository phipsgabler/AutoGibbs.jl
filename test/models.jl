const data_gmm = [
    -1.6518947325446898,
    -0.9012309098240023,
    -0.443220750663359,
]
const s1_gmm = 5.0
const s2_gmm = 1.0

const data_hmm = [
    0.8207426252019316,
    2.081341502733774,
    1.0565299991122412,
]
const s1_hmm = 0.1
const s2_hmm = 0.5

# data for IMM from R. Neal paper: 
const data_neal = [-1.48, -1.40, -1.16, -1.08, -1.02, 0.14, 0.51, 0.53, 0.78]
const α_neal = 10.0
const s1_imm = 5.0
const s2_imm = 1.0


####################################################################################################
####################################################################################################
@model function hierarchical_gaussian(x)
    λ ~ Gamma(2.0, inv(3.0))
    m ~ Normal(0, sqrt(1 / λ))
    x ~ Normal(m, sqrt(1 / λ))
end

@model function bernoulli_mixture(x)
    # Mixture prior.
    w ~ Dirichlet(2, 1/2)

    # Latent probability.
    p ~ DiscreteNonParametric([0.3, 0.7], w)

    # Observation.
    x ~ Bernoulli(p)
end

bernoulli_example(;x = false) = bernoulli_mixture(x)


@model function gmm(x, K)
    N = length(x)

    # Cluster association prior.
    w ~ Dirichlet(K, 1/K)

    # Cluster assignments.
    z ~ filldist(DiscreteNonParametric(1:K, w), N)

    # Cluster centers.
    μ ~ filldist(Normal(0.0, s1_gmm), K)

    # Observations.
    for n = 1:N
        x[n] ~ Normal(μ[z[n]], s2_gmm)
    end

    return x
end

gmm_example(;x = data_gmm, K = 2) = gmm(x, K)
gmm_generate(N; kwargs...) = gmm_generate(Random.GLOBAL_RNG, N; kwargs...)
gmm_generate(rng, N; K = 2) =
    :x => identity.(gmm(fill(missing, N), K)(rng, VarInfo(), SampleFromPrior(), DefaultContext()))


@model function gmm_loopy(x, K, ::Type{T}=Float64) where {T<:Real}
    N = length(x)

    # Cluster centers.
    μ = Vector{T}(undef, K)
    for k = 1:K
        μ[k] ~ Normal(0.0, s2_gmm)
    end

    # Cluster association prior.
    w ~ Dirichlet(K, 1/K)

    # Cluster assignments & observations.
    z = Vector{Int}(undef, N)
    for n = 1:N
        z[n] ~ DiscreteNonParametric(1:K, w)
        x[n] ~ Normal(μ[z[n]], s2_gmm)
    end

    return x
end

gmm_loopy_example(;x = data_gmm, K = 2) = gmm_loopy(x, K)


# same as gmm_loopy, but with an affine transformation on μ.
@model function gmm_shifted(x, K, ::Type{T}=Float64) where {T<:Real}
    N = length(x)

    # Cluster centers.
    μ = Vector{T}(undef, K)
    for k = 1:K
        μ[k] ~ Normal(0.0, s2_gmm)
    end

    # Cluster association prior.
    w ~ Dirichlet(K, 1/K)

    # Cluster assignments & observations.
    z = Vector{Int}(undef, N)
    for n = 1:N
        z[n] ~ DiscreteNonParametric(1:K, w)
        x[n] ~ Normal(4μ[z[n]] - 1, s2_gmm)
    end

    return x
end

gmm_shifted_example(;x = data_gmm, K = 2) = gmm_shifted(x, K)


@model function gmm_marginalized(x, K)
    N = length(x)

    # Cluster association prior.
    w ~ Dirichlet(K, 1/K)

    # Cluster centers.
    μ ~ filldist(Normal(0.0, s1_gmm), K)

    # Observations.
    for n = 1:N
        x[n] ~ MixtureModel(Normal, μ, w)
    end

    return x
end

gmm_marginalized_example(;x = data_gmm, K = 2) = gmm_marginalized(x, K)
gmm_marginalized_generate(N; kwargs...) = gmm_marginalized_generate(Random.GLOBAL_RNG, N; kwargs...)
gmm_marginalized_generate(rng, N; K = 2) =
    :x => identity.(gmm_marginalized(fill(missing, N), K)(rng, VarInfo(), SampleFromPrior(), DefaultContext()))

# K clusters, each one around i for i = 1:K with fixed variance
@model function hmm(x, K, ::Type{X}=Float64) where {X<:Real}
    N = length(x)

    # Transition matrix.
    T = Vector{Vector{X}}(undef, K)
    for i = 1:K
        T[i] ~ Dirichlet(K, 1/K)
    end
    
    # State sequence.
    s = zeros(Int, N)
    s[1] ~ Categorical(K)
    for i = 2:N
        s[i] ~ Categorical(T[s[i-1]])
    end
    
    # Emission matrix.
    m = Vector{T}(undef, K)
    for i = 1:K
        m[i] ~ Normal(i, s1_hmm)
    end
    
    # Observe each point of the input.
    x[1] ~ Normal(m[s[1]], s2_hmm)
    for i = 2:N
        x[i] ~ Normal(m[s[i]], s2_hmm)
    end

    return x
end

hmm_example(;x = data_hmm, K = 2) = hmm(x, K)
hmm_generate(N; kwargs...) = hmm_generate(Random.GLOBAL_RNG, N; kwargs...)
hmm_generate(rng, N; K = 2) =
    :x => identity.(hmm(fill(missing, N), K)(rng, VarInfo(), SampleFromPrior(), DefaultContext()))


@model function changepoint(y)
    N = length(y)
    α = 1 / mean(y)
    λ₁ ~ Exponential(α)
    λ₂ ~ Exponential(α)
    τ ~ DiscreteUniform(1, N)
    
    for n in 1:N
        y[n] ~ Poisson(τ > n ? λ₁ : λ₂)
    end

    return y
end

# @model function reverse_deps(x)
#     m = Vector{Float64}(undef, 2)
#     m[1] ~ Normal()
#     m[2] ~ Normal()
#     x ~ MvNormal(m)
# end


function stickbreak(v)
    K = length(v) + 1
    cumprod_one_minus_v = cumprod(1 .- v)

    return map(1:K) do k
        if k == 1
            v[1]
        elseif k == K
            cumprod_one_minus_v[K - 1]
        else
            v[k] * cumprod_one_minus_v[k - 1]
        end
    end
end

# from https://luiarthur.github.io/TuringBnpBenchmarks/dpsbgmm
@model function imm_stick(y, α, K)
    N = length(y)
    crm = DirichletProcess(α)

    v ~ filldist(StickBreakingProcess(crm), K - 1)
    w = stickbreak(v)
    
    # Cluster assignments
    z = zeros(Int, N)
    for n = 1:N
        z[n] ~ Categorical(w)
    end

    # Cluster centers
    L = identity(K)
    μ ~ filldist(Normal(0.0, s1_imm), L)

    # Observations
    for n = 1:N
        y[n] ~ Normal(μ[z[n]], s2_imm)
    end

    return y
end

imm_stick_example(;y = data_neal, α = α_neal, K = 10) = imm_stick(y, α, K)
imm_stick_generate(N; kwargs...) = imm_stick_generate(Random.GLOBAL_RNG, N; kwargs...)
imm_stick_generate(rng, N; α = α_neal, K = 10) =
    :y => identity.(imm_stick(fill(missing, N), α, K)(rng, VarInfo(), SampleFromPrior(), DefaultContext()))






####################################################################################################
####################################################################################################
@model function gmm_tarray(x, K)
    N = length(x)

    # Cluster association prior.
    w ~ Dirichlet(K, 1/K)

    # Cluster assignments.
    z = tzeros(Int, N)
    for n = 1:N
        z[n] ~ DiscreteNonParametric(1:K, w)
    end
    
    # Cluster centers.
    μ ~ filldist(Normal(0.0, s1_gmm), K)

    # Observations.
    for n = 1:N
        x[n] ~ Normal(μ[z[n]], s2_gmm)
    end

    return x
end

gmm_tarray_example(;x = data_gmm, K = 2) = gmm_tarray(x, K)


# K clusters, each one around i for i = 1:K with fixed variance
@model function hmm_tarray(x, K)
    N = length(x)

    # Transition matrix.
    T ~ filldist(Dirichlet(K, 1/K), K)
    
    # State sequence.
    s = tzeros(Int, N)
    s[1] ~ Categorical(K)
    for i = 2:N
        s[i] ~ Categorical(T[:,s[i-1]])
    end
    
    # Emission matrix.
    m ~ arraydist(Normal(1:K, s1_hmm))

    # Observe each point of the input.
    x[1] ~ Normal(m[s[1]], s2_hmm)
    for i = 2:N
        x[i] ~ Normal(m[s[i]], s2_hmm)
    end

    return x
end

hmm_tarray_example(;x = data_hmm, K = 2) = hmm_tarray(x, K)


function stickbreak(v)
    K = length(v) + 1
    cumprod_one_minus_v = cumprod(1 .- v)

    return map(1:K) do k
        if k == 1
            v[1]
        elseif k == K
            cumprod_one_minus_v[K - 1]
        else
            v[k] * cumprod_one_minus_v[k - 1]
        end
    end
end

@model function imm_stick_tarray(y, α, K)
    N = length(y)
    crm = DirichletProcess(α)

    v ~ filldist(StickBreakingProcess(crm), K - 1)
    w = stickbreak(v)
    
    # Cluster assignments
    z = tzeros(Int, N)
    for n = 1:N
        z[n] ~ Categorical(w)
    end

    # Cluster centers
    L = identity(K)
    μ ~ filldist(Normal(0.0, s1_imm), L)

    # Observations
    for n = 1:N
        y[n] ~ Normal(μ[z[n]], s2_imm)
    end

    return y
end

imm_stick_tarray_example(;y = data_neal, α = α_neal, K = 10) = imm_stick_tarray(y, α, K)
