@model function bernoulli_mixture(x)
    w ~ Dirichlet(2, 1.0)
    p ~ DiscreteNonParametric([0.3, 0.7], w)
    x ~ Bernoulli(p)
end

model_bernoulli = bernoulli_mixture(false)
graph_bernoulli = trackdependencies(model_bernoulli)
@testdependencies(model_bernoulli, w, p, x)


@model function gmm(x, K)
    N = length(x)

    μ ~ filldist(Normal(), K)
    w ~ Dirichlet(K, 1.0)
    z ~ filldist(Categorical(w), N)

    for n in eachindex(x)
        x[n] ~ Normal(μ[z[n]], 1.0)
    end
end

model_gmm = gmm([0.1, -0.05, 1.0], 2)
graph_gmm = trackdependencies(model_gmm)
@testdependencies(model_gmm, μ, w, z, x[1], x[2], x[3])


@model function hmm(x, K)
    # Get observation length.
    N = length(x)

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
    x[1] ~ Normal(m[s[1]], 0.1)

    for i = 2:N
        s[i] ~ Categorical(T[s[i-1]])
        x[i] ~ Normal(m[s[i]], 0.1)
    end
end

model_hmm = hmm([0.1, -0.05, 1.0], 2)
graph_hmm = trackdependencies(model_hmm)
@testdependencies(model_hmm, T[1], T[2], m[1], m[2], s[1], s[2], s[3], x[1], x[2], x[3])


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

    x = 
    for n in 1:N
        x[n] ~ Normal(μ[z[n]], 1.0)
    end
end

model_imm = imm([0.1, -0.05, 1.0])
graph_imm = trackdependencies(model_imm)
@testdependencies(model_imm, z[1], z[2], z[3], μ, x[1], x[2], x[3])


@model function changepoint(y)
    α = 1/mean(y)
    λ1 ~ Exponential(α)
    λ2 ~ Exponential(α)
    τ ~ DiscreteUniform(1, length(y))
    for idx in 1:length(y)
        y[idx] ~ Poisson(τ > idx ? λ1 : λ2)
    end
end

model_changepoint = changepoint([1.1, 0.9, 0.2])
graph_changepoint = trackdependencies(model_changepoint)
@testdependencies(model_changepoint, λ1, λ2, τ, y[1], y[2], y[3])
