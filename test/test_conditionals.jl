@model function bernoulli_mixture(x)
    w ~ Dirichlet(2, 1.0)
    p ~ DiscreteNonParametric([0.3, 0.7], w)
    x ~ Bernoulli(p)
end

model_bernoulli = bernoulli_mixture(false)
graph_bernoulli = trackdependencies(model_bernoulli)
@testdependencies(model_bernoulli, w, p, x)
@test_nothrow sample(model_bernoulli, Gibbs(AutoConditional(:p), MH(:w)), 2)


let w = graph_bernoulli[4].value,
    p = graph_bernoulli[6].value,
    x = graph_bernoulli[2].value,
    p1 = w[1] * pdf(Bernoulli(p), x),
    p2 = w[2] * pdf(Bernoulli(p), x),
    Z = p1 + p2

    # 𝓅(p | w, x) ∝ 𝓅(p | w) * 𝓅(x | p)
    D_cond = DiscreteNonParametric([0.3, 0.7], [p1 / Z, p2 / Z])
    
    local conditional
    @test_nothrow conditional = conditional_dists(graph_bernoulli, @varname(p))[@varname(p)]
    @test issimilar(conditional, D_cond)
end


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
@test_nothrow sample(model_gmm, Gibbs(AutoConditional(:z), MH(:w, :μ)), 2)
@test_nothrow sample(model_gmm, Gibbs(AutoConditional(:z), HMC(0.01, 10, :w, :μ)), 2)

let μ = graph_gmm[7].value,
    w = graph_gmm[9].value,
    z = graph_gmm[12].value,
    x = graph_gmm[2].value,
    p_z1 = w[1] .* pdf.(Normal.(μ[z], 1.0), x),
    p_z2 = w[2] .* pdf.(Normal.(μ[z], 1.0), x),
    Z = p_z1 .+ p_z2

    # 𝓅(zᵢ | μ, w, x, z₋ᵢ) ∝ 𝓅(zᵢ | w) * 𝓅(xᵢ | zᵢ, μ)
    D_cond = Product([DiscreteNonParametric([1, 2], [p1, p2])
                      for (p1, p2) in  zip(p_z1 ./ Z, p_z2 ./ Z)])

    local conditional
    @test_nothrow conditional = conditional_dists(graph_gmm, @varname(z))[@varname(z)]
    @test issimilar(conditional, D_cond)
end


@model function gmm_loopy(x, K, ::Type{T}=Float64) where {T<:Real}
    N = length(x)

    μ = Vector{T}(undef, K)
    for k = 1:K
        μ[k] ~ Normal()
    end
    
    w ~ Dirichlet(K, 1.0)
    z = Vector{Int}(undef, N)
    for n = 1:N
        z[n] ~ Categorical(w)
        x[n] ~ Normal(μ[z[n]], 1.0)
    end
end

model_gmm_loopy = gmm_loopy([0.1, -0.05, 1.0], 2)
graph_gmm_loopy = trackdependencies(model_gmm_loopy)
@testdependencies(model_gmm_loopy, μ[1], μ[2], w, z[1], z[2], z[3], x[1], x[2], x[3])
@test_nothrow sample(model_gmm_loopy, Gibbs(AutoConditional(:z), MH(:w, :μ)), 2)
@test_nothrow sample(model_gmm_loopy, Gibbs(AutoConditional(:z), HMC(0.01, 10, :w, :μ)), 2)

let μ = [graph_gmm_loopy[19].value, graph_gmm_loopy[28].value],
    w = graph_gmm_loopy[31].value,
    z = [graph_gmm_loopy[42].value, graph_gmm_loopy[60].value, graph_gmm_loopy[77].value],
    x = graph_gmm_loopy[2].value,
    p_z1 = w[1] .* pdf.(Normal.(μ[z], 1.0), x),
    p_z2 = w[2] .* pdf.(Normal.(μ[z], 1.0), x),
    Z = p_z1 .+ p_z2

    # 𝓅(zᵢ | μ, w, x, z₋ᵢ) ∝ 𝓅(zᵢ | w) * 𝓅(xᵢ | zᵢ, μ)
    D_conds = Dict(@varname(z[i]) => DiscreteNonParametric([1, 2], [p1, p2])
               for (i, p1, p2) in  zip(1:3, p_z1 ./ Z, p_z2 ./ Z))

    local conditionals
    @test_nothrow conditionals = conditional_dists(graph_gmm_loopy, @varname(z))
    for (vn, conditional) in conditionals
        @show probs(conditional), probs(D_conds[vn])
        @test issimilar(conditional, D_conds[vn])
    end
end


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

model_hmm = hmm([0.1, -0.05, 1.0], 2)
graph_hmm = trackdependencies(model_hmm)
@testdependencies(model_hmm, T[1], T[2], m[1], m[2], s[1], s[2], s[3], x[1], x[2], x[3])
@test_nothrow sample(model_hmm, Gibbs(AutoConditional(:s), MH(:m, :T)), 2)
@test_nothrow sample(model_hmm, Gibbs(AutoConditional(:s), HMC(0.01, 10, :m, :T)), 2)


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


model_imm = imm([0.1, -0.05, 1.0])
graph_imm = trackdependencies(model_imm)
@testdependencies(model_imm, z[1], z[2], z[3], μ, x[1], x[2], x[3])
@test_nothrow sample(model_imm, Gibbs(AutoConditional(:z), MH(:μ)), 2)
@test_nothrow sample(model_imm, Gibbs(AutoConditional(:z), HMC(0.01, 10, :μ)), 2)


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
@test_nothrow sample(model_changepoint, Gibbs(AutoConditional(:τ), MH(:λ1, :λ2)), 2)


@model function reverse_deps(x)
    m = Vector{Float64}(undef, 2)
    m[1] ~ Normal()
    m[2] ~ Normal()
    x ~ MvNormal(m)
end

model_reverse_deps = reverse_deps([0.1, -0.2])
graph_reverse_deps = trackdependencies(model_reverse_deps)
@testdependencies(model_reverse_deps, m[1], m[2], x)
