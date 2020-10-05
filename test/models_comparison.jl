@model function gmm_tarray(x, K)
    N = length(x)
    
    # Cluster centers.
    μ ~ filldist(Normal(), K)

    # Cluster association prior.
    w ~ Dirichlet(K, 1.0)

    # Cluster assignments.
    z = tzeros(Int, N)
    for n = 1:N
        z[n] ~ DiscreteNonParametric(1:K, w)
    end
    
    # Observations.
    for n = 1:N
        x[n] ~ Normal(μ[z[n]], 1.0)
    end

    return x
end

gmm_tarray_example(;x = [0.1, -0.05, 1.0], K = 2) = gmm_tarray(x, K)
gmm_generate(N; K = 2) = identity.(gmm_tarray(fill(missing, N), K)())

# K clusters, each one around i for i = 1:K with variance 0.5
@model function hmm_tarray(x, K, ::Type{T}=Float64) where {T<:Real}
    N = length(x)

    # State sequence.
    s = tzeros(Int, N)

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

    return x
end

hmm_tarray_example(x = [0.1, -0.05, 1.0], K = 2) = hmm_tarray(x, K)
hmm_generate(N; K = 2) = identity.(hmm_tarray(fill(missing, N), K)())

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
    μ ~ filldist(Normal(), L)

    # Observations
    for n = 1:N
        y[n] ~ Normal(μ[z[n]], 1.0)
    end

    return y
end

imm_stick_tarray_example(;y = data_neal, α = α_neal, K = 10) = imm_stick_tarray()
imm_stick_generate(N; α = α_neal, K = 10) = identity.(hmm_tarray(fill(missing, N), α, K)())
