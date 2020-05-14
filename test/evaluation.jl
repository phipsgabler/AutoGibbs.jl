@model function gmm(x, K)
    N = length(x)

    μ ~ filldist(Normal(), K)
    w ~ Dirichlet(K, 1.0)
    z ~ filldist(Categorical(w), N)

    for n in eachindex(x)
        x[n] ~ Normal(μ[z[n]], 1.0)
    end
end

graph_hmm = trackdependencies(gmm([0.1, -0.05, 1.0], 2))


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
        T[i] ~ Dirichlet(ones(K)/K)
        m[i] ~ Normal(i, 0.5)
    end
    
    # Observe each point of the input.
    s[1] ~ Categorical(K)
    y[1] ~ Normal(m[s[1]], 0.1)

    for i = 2:N
        s[i] ~ Categorical(vec(T[s[i-1]]))
        y[i] ~ Normal(m[s[i]], 0.1)
    end
end

graph_hmm = trackdependencies(imm([0.1, -0.05, 1.0], 2))


@model function imm(x)
    N = length(x)

    nk = zeros(Int, N)
    z = zeros(Int, length(x))
    for i in eachindex(x)
        z[i] ~ ChineseRestaurantProcess(DirichletProcess(1.0), nk)
        nk[z[i]] += 1
    end

    K = findlast(!iszero, nk)
    μ ~ filldist(Normal(), K)

    for n in eachindex(x)
        x[n] ~ Normal(μ[z[n]], 1.0)
    end
end

graph_imm = trackdependencies(imm([0.1, -0.05, 1.0]))
