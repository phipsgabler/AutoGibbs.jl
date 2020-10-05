@model function bernoulli_mixture(x)
    # Mixture prior.
    w ~ Dirichlet(2, 1.0)

    # Latent probability.
    p ~ DiscreteNonParametric([0.3, 0.7], w)

    # Observation.
    x ~ Bernoulli(p)
end

bernoulli_example(;x = false) = bernoulli_mixture(x)


@model function gmm(x, K)
    N = length(x)
    
    # Cluster centers.
    μ ~ filldist(Normal(), K)

    # Cluster association prior.
    w ~ Dirichlet(K, 1.0)

    # Cluster assignments.
    z ~ filldist(DiscreteNonParametric(1:K, w), N)

    # Observations.
    for n = 1:N
        x[n] ~ Normal(μ[z[n]], 1.0)
    end
end

gmm_example(;x = [0.1, -0.05, 1.0], K = 2) = gmm(x, K)


@model function gmm_loopy(x, K, ::Type{T}=Float64) where {T<:Real}
    N = length(x)

    # Cluster centers.
    μ = Vector{T}(undef, K)
    for k = 1:K
        μ[k] ~ Normal()
    end

    # Cluster association prior.
    w ~ Dirichlet(K, 1.0)

    # Cluster assignments & observations.
    z = Vector{Int}(undef, N)
    for n = 1:N
        z[n] ~ DiscreteNonParametric(1:K, w)
        x[n] ~ Normal(μ[z[n]], 1.0)
    end
end

# same as gmm_loopy, but with an affine transformation on μ.
@model function gmm_shifted(x, K, ::Type{T}=Float64) where {T<:Real}
    N = length(x)

    # Cluster centers.
    μ = Vector{T}(undef, K)
    for k = 1:K
        μ[k] ~ Normal()
    end

    # Cluster association prior.
    w ~ Dirichlet(K, 1.0)

    # Cluster assignments & observations.
    z = Vector{Int}(undef, N)
    for n = 1:N
        z[n] ~ DiscreteNonParametric(1:K, w)
        x[n] ~ Normal(4μ[z[n]] - 1, 1.0)
    end
end

# K clusters, each one around i for i = 1:K with variance 0.5
@model function hmm(x, K, ::Type{T}=Float64) where {T<:Real}
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

hmm_example(;x = [1.1, 0.95, 2.2], K = 2) = hmm(x, K)


@model function changepoint(y)
    N = length(y)
    α = 1 / mean(y)
    λ₁ ~ Exponential(α)
    λ₂ ~ Exponential(α)
    τ ~ DiscreteUniform(1, N)
    
    for n in 1:N
        y[n] ~ Poisson(τ > n ? λ₁ : λ₂)
    end
end

# @model function reverse_deps(x)
#     m = Vector{Float64}(undef, 2)
#     m[1] ~ Normal()
#     m[2] ~ Normal()
#     x ~ MvNormal(m)
# end


###########################################################################
# data from R. Neal paper: 
const data_neal = [-1.48, -1.40, -1.16, -1.08, -1.02, 0.14, 0.51, 0.53, 0.78]
const α_neal = 10.0

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
    μ ~ filldist(Normal(), L)

    # Observations
    for n = 1:N
        y[n] ~ Normal(μ[z[n]], 1.0)
    end
end

imm_stick_example(;y = data_neal, α = α_neal, K = 10) = imm_stick(y, α, K)


