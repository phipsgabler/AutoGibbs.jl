@model function bernoulli_mixture(x)
    w ~ Dirichlet(2, 1.0)
    p ~ DiscreteNonParametric([0.3, 0.7], w)
    x ~ Bernoulli(p)
end

graph_bernoulli = trackdependencies(bernoulli_mixture(false))

@model function gmm(x, K)
    N = length(x)

    μ ~ filldist(Normal(), K)
    w ~ Dirichlet(K, 1.0)
    z ~ filldist(Categorical(w), N)

    for n in eachindex(x)
        x[n] ~ Normal(μ[z[n]], 1.0)
    end
end

graph_gmm = trackdependencies(gmm([0.1, -0.05, 1.0], 2))


@model function hmm(y, K)
    # Get observation length.
    N = length(y)

    # State sequence.
    s = zeros(Int, N)

    # Emission matrix.
    m = Vector(undef, K)

    # Transition matrix.
    T = Vector{Vector}(undef, K)

    # Assign distributions to each element
    # of the transition matrix and the
    # emission matrix.
    for i = 1:K
        T[i] ~ Dirichlet(K, 1.0)
        m[i] ~ Normal(i, 0.5)
    end
    
    # Observe each point of the input.
    s[1] ~ Categorical(K)
    y[1] ~ Normal(m[s[1]], 0.1)

    for i = 2:N
        s[i] ~ Categorical(T[s[i-1]])
        y[i] ~ Normal(m[s[i]], 0.1)
    end
end

graph_hmm = trackdependencies(hmm([0.1, -0.05, 1.0], 2))


@model function imm(x)
    N = length(x)

    nk = zeros(Int, N)
    G = ChineseRestaurantProcess(DirichletProcess(1.0), nk)
    
    z = zeros(Int, length(x))
    for i in eachindex(x)
        z[i] ~ G
        nk[z[i]] += 1
    end

    K = findlast(!iszero, nk)
    μ ~ filldist(Normal(), K)

    for n in eachindex(x)
        x[n] ~ Normal(μ[z[n]], 1.0)
    end
end

graph_imm = trackdependencies(imm([0.1, -0.05, 1.0]))


@model function changepoint(y)
    α = 1/mean(y)
    λ1 ~ Exponential(α)
    λ2 ~ Exponential(α)
    τ ~ DiscreteUniform(1, length(y))
    for idx in 1:length(y)
        y[idx] ~ Poisson(τ > idx ? λ1 : λ2)
    end
end

graph_changepoint = trackdependencies(changepoint([1.1, 0.9, 0.2, 0.3]))
