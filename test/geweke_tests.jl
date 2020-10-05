const K = 2
const N = 10
const α_neal = 10.0

@model function gmm_joint(z, x, ::Type{T}=Float64) where T
    if z === missing
        z = Vector{Int}(undef, N)
    end

    if x === missing
        x = Vector{T}(undef, N)
    end
    
    
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

    return z, x
end


@model function hmm_joint(s, x, ::Type{T}=Float64) where {T<:Real}
    if s === missing
        s = Vector{Int}(undef, N)
    end

    if x === missing
        x = Vector{T}(undef, N)
    end

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

    return s, x
end


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
@model function imm_stick_joint(z, y, ::Type{T}=Float64) where {T}
    if z === missing
        z = Vector{Int}(undef, N)
    end

    if x === missing
        x = Vector{T}(undef, N)
    end
    
    crm = DirichletProcess(α_neal)

    v ~ filldist(StickBreakingProcess(crm), K - 1)
    w = stickbreak(v)
    
    # Cluster assignments
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


# test function
g(θ, x) = cat(θ, x; dims=1)

function geweke_gmm()
    rand_θ_given(x) = rand(StaticConditional(gmm_joint(missing, x), :z))

    test = perform(GewekeTest(2_000), gmm_joint, rand_θ_given, g)
    # plot(test, gmm_joint; size=(300, 300), title="GMM Geweke statistic")
    s = compute_statistic!(test, g)
    @test s.pvalue < 0.05
end
