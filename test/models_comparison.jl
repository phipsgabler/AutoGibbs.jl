@model function gmm_tarray(x, K)
    N = length(x)

    # Cluster centers.
    μ ~ filldist(Normal(), K)

    # Cluster association prior.
    w ~ Dirichlet(K, 1.0)

    # Cluster assignments.
    z = tzeros(Int, N)

    # Observations.
    for n = 1:N
        z[n] ~ DiscreteNonParametric(1:K, w)
        x[n] ~ Normal(μ[z[n]], 1.0)
    end
end

gmm_tarray_example(x = [0.1, -0.05, 1.0], K = 2) = gmm_tarray(x, K)


# K clusters, each one around i for i = 1:K with variance 0.5
@model function hmm_tarray(x, K, ::Type{T}=Float64) where {T<:Real}
    N = length(x)

    # State sequence.
    s = tzeros(Int, N)

    # Emission matrix.
    m ~ arraydist([Normal(i, 0.5) for i in 1:K])

    # Transition matrix.
    t ~ filldist(Dirichlet(K, 1.0), K)

    # Observe each point of the input.
    s[1] ~ Categorical(K)
    x[1] ~ Normal(m[s[1]], 0.1)

    for i = 2:N
        s[i] ~ Categorical(t[:,s[i-1]])
        x[i] ~ Normal(m[s[i]], 0.1)
    end
end

hmm_tarray_example(x = [0.1, -0.05, 1.0], K = 2) = hmm_tarray(x, K)


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

    # Cluster centers
    L = identity(K) # Is this needed?
    μ ~ filldist(Normal(), L)

    # Cluster assignments
    z = tzeros(Int, N)

    # Observations
    for n = 1:N
        z[n] ~ Categorical(w)
        y[n] ~ Normal(μ[z[n]], 1.0)
    end
end

imm_stick_tarray_example(y = data_neal, α = α_neal, K = 100) = imm_stick_tarray(y, α, K)
