@model function bernoulli_mixture(x)
    # Mixture prior.
    w ~ Dirichlet(2, 1.0)

    # Latent probability.
    p ~ DiscreteNonParametric([0.3, 0.7], w)

    # Observation.
    x ~ Bernoulli(p)
end

function test_bernoulli()
    model_bernoulli = bernoulli_mixture(false)
    graph_bernoulli = trackdependencies(model_bernoulli)
    @testdependencies(model_bernoulli, w, p, x)
    cond_bernoulli_p = StaticConditional(model_bernoulli, :p)
    @test_nothrow sample(model_bernoulli, Gibbs(cond_bernoulli_p, MH(:w)), 500)


    # Analytic tests
    w = graph_bernoulli[4].value
    p = graph_bernoulli[6].value
    x = graph_bernoulli[2].value
    p_1 = w[1] * pdf(Bernoulli(0.3), x)
    p_2 = w[2] * pdf(Bernoulli(0.7), x)
    Z = p_1 + p_2

    # 𝓅(p | w, x) ∝ 𝓅(p | w) * 𝓅(x | p)
    analytic_conditional = DiscreteNonParametric([0.3, 0.7], [p_1 / Z, p_2 / Z])
    @info "Bernoulli analytic" analytic_conditional
    θ = AutoGibbs.sampled_values(graph_bernoulli)
    
    local calculated_conditional
    @test_nothrow calculated_conditional = conditionals(graph_bernoulli, @varname(p))[@varname(p)]
    @info "Bernoulli calculated" calculated_conditional(θ)
    
    @test issimilar(calculated_conditional(θ), analytic_conditional)
end


###########################################################################
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

function test_gmm()
    model_gmm = gmm([0.1, -0.05, 1.0], 2)
    graph_gmm = trackdependencies(model_gmm)
    @testdependencies(model_gmm, μ, w, z, x[1], x[2], x[3])
    cond_gmm_z = StaticConditional(model_gmm, :z)
    @test_nothrow sample(model_gmm, Gibbs(cond_gmm_z, MH(:w, :μ)), 500)
    @test_nothrow sample(model_gmm, Gibbs(cond_gmm_z, HMC(0.01, 10, :w, :μ)), 500)


    # Analytic tests
    μ = graph_gmm[7].value
    w = graph_gmm[9].value
    z = graph_gmm[13].value
    x = graph_gmm[2].value
    p_1 = w[1] .* pdf.(Normal(μ[1], 1.0), x)
    p_2 = w[2] .* pdf.(Normal(μ[2], 1.0), x)
    (Z1, Z2, Z3) = p_1 .+ p_2

    # 𝓅(zᵢ | μ, w, x, z₋ᵢ) ∝ 𝓅(zᵢ | w) * 𝓅(xᵢ | zᵢ, μ)
    analytic_conditionals = [@varname(z[1]) => Categorical([p_1[1], p_2[1]] ./ Z1),
                             @varname(z[2]) => Categorical([p_1[2], p_2[2]] ./ Z2),
                             @varname(z[3]) => Categorical([p_1[3], p_2[3]] ./ Z3)]
    @info "GMM analytic" analytic_conditionals
    θ = AutoGibbs.sampled_values(graph_gmm)

    local calculated_conditional
    @test_nothrow calculated_conditional = conditionals(graph_gmm, @varname(z))[@varname(z)]
    @info "GMM calculated" calculated_conditional(θ)
    
    @test issimilar(calculated_conditional(θ), Product([D for (vn, D) in analytic_conditionals]))
end


###########################################################################
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

function test_gmm_loopy()
    model_gmm_loopy = gmm_loopy([0.1, -0.05, 1.0], 2)
    graph_gmm_loopy = trackdependencies(model_gmm_loopy)
    @testdependencies(model_gmm_loopy, μ[1], μ[2], w, z[1], z[2], z[3], x[1], x[2], x[3])
    cond_gmm_loopy_z = StaticConditional(model_gmm_loopy, :z)
    @test_nothrow sample(model_gmm_loopy, Gibbs(cond_gmm_loopy_z, MH(:w, :μ)), 500)
    @test_nothrow sample(model_gmm_loopy, Gibbs(cond_gmm_loopy_z, HMC(0.01, 10, :w, :μ)), 500)


    # Analytic tests
    μ = [graph_gmm_loopy[19].value, graph_gmm_loopy[28].value]
    w = graph_gmm_loopy[31].value
    z = [graph_gmm_loopy[43].value, graph_gmm_loopy[62].value, graph_gmm_loopy[80].value]
    x = graph_gmm_loopy[2].value
    p_1 = w[1] .* pdf.(Normal(μ[1], 1.0), x)
    p_2 = w[2] .* pdf.(Normal(μ[2], 1.0), x)
    (Z1, Z2, Z3) = p_1 .+ p_2

    # 𝓅(zᵢ | μ, w, x, z₋ᵢ) ∝ 𝓅(zᵢ | w) * 𝓅(xᵢ | zᵢ, μ)
    analytic_conditionals = [@varname(z[1]) => Categorical([p_1[1], p_2[1]] ./ Z1),
                             @varname(z[2]) => Categorical([p_1[2], p_2[2]] ./ Z2),
                             @varname(z[3]) => Categorical([p_1[3], p_2[3]] ./ Z3)]
    @info "Loopy GMM analytic" analytic_conditionals
    θ = AutoGibbs.sampled_values(graph_gmm_loopy)
    
    local calculated_conditionals
    @test_nothrow calculated_conditionals = conditionals(graph_gmm_loopy, @varname(z))
    @info "Loopy GMM calculated" Dict(vn => cond(θ) for (vn, cond) in calculated_conditionals)
    
    for (vn, analytic_conditional) in analytic_conditionals
        @test issimilar(calculated_conditionals[vn](θ), analytic_conditional)
    end
end


###########################################################################
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

function test_gmm_shifted()
    model_gmm_shifted = gmm_shifted([0.1, -0.05, 1.0], 2)
    graph_gmm_shifted = trackdependencies(model_gmm_shifted)
    @testdependencies(model_gmm_shifted, μ[1], μ[2], w, z[1], z[2], z[3], x[1], x[2], x[3])
    cond_gmm_shifted_z = StaticConditional(model_gmm_shifted, :z)
    @test_nothrow sample(model_gmm_shifted, Gibbs(cond_gmm_shifted_z, MH(:w, :μ)), 500)
    @test_nothrow sample(model_gmm_shifted, Gibbs(cond_gmm_shifted_z, HMC(0.01, 10, :w, :μ)), 500)
end


###########################################################################
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


function test_hmm()
    model_hmm = hmm([1.1, 0.95, 2.2], 2)
    graph_hmm = trackdependencies(model_hmm)
    @testdependencies(model_hmm, T[1], T[2], m[1], m[2], s[1], s[2], s[3], x[1], x[2], x[3])
    cond_hmm_s = StaticConditional(model_hmm, :s)
    @test_nothrow sample(model_hmm, Gibbs(cond_hmm_s, MH(:m, :T)), 500)
    @test_nothrow sample(model_hmm, Gibbs(cond_hmm_s, HMC(0.01, 10, :m, :T)), 500)


    # Analytic tests
    T = [graph_hmm[23].value, graph_hmm[38].value]
    m = [graph_hmm[29].value, graph_hmm[44].value]
    s1 = graph_hmm[48].value
    s2 = graph_hmm[68].value
    s3 = graph_hmm[88].value
    x = graph_hmm[2].value
    D_obs_1 = Normal(m[1], 0.1)
    D_obs_2 = Normal(m[2], 0.1)
    p_s1_1 = pdf(Categorical(2), 1) * pdf(Categorical(T[1]), s2) * pdf(D_obs_1, x[1])
    p_s1_2 = pdf(Categorical(2), 2) * pdf(Categorical(T[2]), s2) * pdf(D_obs_2, x[1])
    p_s2_1 = pdf(Categorical(T[s1]), 1) * pdf(Categorical(T[1]), s3) * pdf(D_obs_1, x[2])
    p_s2_2 = pdf(Categorical(T[s1]), 2) * pdf(Categorical(T[2]), s3) * pdf(D_obs_2, x[2])
    p_s3_1 = pdf(Categorical(T[s2]), 1) * pdf(D_obs_1, x[3])
    p_s3_2 = pdf(Categorical(T[s2]), 2) * pdf(D_obs_2, x[3])
    Z_1 = p_s1_1 + p_s1_2
    Z_2 = p_s2_1 + p_s2_2
    Z_3 = p_s3_1 + p_s3_2

    # 𝓅(s₁ | T, m, s₋₁, x) ∝ 𝓅(s₁) 𝓅(s₂ | s₁, T) 𝓅(x₁ | s₁, m)
    #  𝓅(sᵢ | T, m, s₋ᵢ, x) ∝  𝓅(sᵢ | sᵢ₋₁, T) 𝓅(sᵢ₊₁ | sᵢ, T) 𝓅(xᵢ | sᵢ, m) (for i ≥ 2)
    analytic_conditionals = [@varname(s[1]) => Categorical([p_s1_1, p_s1_2] ./ Z_1),
                             @varname(s[2]) => Categorical([p_s2_1, p_s2_2] ./ Z_2),
                             @varname(s[3]) => Categorical([p_s3_1, p_s3_2] ./ Z_3)]
    θ = AutoGibbs.sampled_values(graph_hmm)
    @info "HMM analytic" analytic_conditionals
    
    local calculated_conditionals
    @test_nothrow calculated_conditionals = conditionals(graph_hmm, @varname(s))
    @info "HMM calculated" Dict(vn => cond(θ) for (vn, cond) in calculated_conditionals)
    
    for (vn, analytic_conditional) in analytic_conditionals
        # @show vn => probs(calculated_conditionals[vn]), probs(analytic_conditional)
        @test issimilar(calculated_conditionals[vn](θ), analytic_conditional)
    end
end


###########################################################################
function update_histogram!(histogram, bin)
    if bin > length(histogram)
        push!(histogram, 1)
    else
        histogram[bin] += 1
    end

    return histogram
end


Turing.RandomMeasures.DirichletProcess(α, G₀) = DirichletProcess(α)


@model function imm(y, α, ::Type{T}=Vector{Float64}) where {T}
    N = length(y)

    K = 0
    nk = Vector{Int}()
    z = Vector{Int}(undef, N)

    G₀ = Normal()
    
    for n = 1:N
        z[n] ~ ChineseRestaurantProcess(DirichletProcess(α, G₀), nk)
        nk = update_histogram!(nk, z[n])
        K = max(K, z[n])
    end

    μ = T(undef, K)
    for k = 1:K
        μ[k] ~ G₀
    end
    
    for n = 1:N
        y[n] ~ Normal(μ[z[n]], 1.0)
    end
end

function test_imm()
    # data from R. Neal paper: [-1.48, -1.40, -1.16, -1.08, -1.02, 0.14, 0.51, 0.53, 0.78]
    model_imm = imm([-1.02, 0.14, 0.51], 10.0)
    graph_imm = trackdependencies(model_imm)
    # we leave out the μs, because there might be 1--3 of them
    @testdependencies(model_imm, z[1], z[2], z[3], y[1], y[2], y[3])
    @test_nothrow sample(model_imm, Gibbs(AutoConditional(:z), HMC(0.01, 10, :μ)), 2)

    # for comparison:
    # sample(model_imm, Gibbs(MH(:z => filldist(Categorical(9), 9)), HMC(0.01, 10, :μ)), 2)


    # Analytic tests
    z = [graph_imm[19].value, graph_imm[36].value, graph_imm[52].value]
    μ = [v.value for v in values(graph_imm.statements)
         if v isa AutoGibbs.Assumption && DynamicPPL.subsumes(@varname(z), v.vn)]
    y = graph_imm[2].value
    K = graph_imm[55].value
    N = graph_imm[7].value

    CRP(h) = ChineseRestaurantProcess(DirichletProcess(1.0), h)
    _pdf(d, x) = exp(logpdf(d, x))

    # 𝓅(zₙ = k| z₁, ..., zₙ₋₁, μ, yₙ) ∝ (∏_{i = z ≥ n} 𝓅(zᵢ | z₁,...,zᵢ)) 𝓅(yₙ | zₙ, μ)
    function cond(n, k)
        # 𝓅(zₙ = k | z₁, ..., zₙ₋₁)
        l = _pdf(CRP(z[1:n-1]), k)

        # 𝓅(zₙ₊ᵢ | z₁, ..., zₙ₊ᵢ₋₁) for i = n+1 .. N
        for i = n+1:N
            l += _pdf(CRP([j == n ? k : z[j] for j = 1:i-1]), z[i])
        end

        if k <= K
            # 𝓅(yₙ | zₙ = k, μ)
            l += pdf(Normal(μ[k]), y[n])
        else
            # 𝓅(yₙ | zₙ = K + 1, μ) = ∫ 𝓅(yₙ | m) 𝓅(m) dm
            m = rand(Normal(), 100)
            l += mean(pdf.(Normal.(m), y[n]))
        end

        return l
    end
    
    p_z1_1 = cond(1, 1)
    p_z2_1, p_z2_2 = cond(2, 1), cond(2, 2)
    p_z3_1, p_z3_2, p_z3_3 = cond(3, 1), cond(3, 2), cond(3, 3)
    Z_1 = p_z1_1
    Z_2 = p_z2_1 + p_z2_2
    Z_3 = p_z3_1 + p_z3_2 + p_z3_3
    
    analytic_conditionals = [@varname(z[1]) => Categorical([p_z1_1] ./ Z_1),
                             @varname(z[2]) => Categorical([p_z2_1, p_z2_2] ./ Z_2),
                             @varname(z[3]) => Categorical([p_z3_1, p_z3_2, p_z3_3] ./ Z_3)]
    θ = AutoGibbs.sampled_values(graph_imm)
    @info "IMM analytic" analytic_conditionals
    
    local calculated_conditionals
    @test_nothrow calculated_conditionals = conditionals(graph_imm, @varname(z))
    @info "IMM calculated" Dict(vn => cond(θ) for (vn, cond) in calculated_conditionals)
    
    # for (vn, analytic_conditional) in analytic_conditionals
    #     # @show vn => probs(calculated_conditionals[vn]), probs(analytic_conditional)
    #     @test issimilar(calculated_conditionals[vn](θ), analytic_conditional)
    # end
end

###########################################################################
@model function changepoint(y)
    N = length(y)
    α = 1 / mean(y)
    λ₁ ~ Exponential(α)
    λ₂ ~ Exponential(α)
    τ ~ DiscreteUniform(1, N)
    
    for n in 1:N
        y[n] ~ Poisson(τ > N ? λ₁ : λ₂)
    end
end

function test_changepoint()
    model_changepoint = changepoint([1.1, 0.9, 0.2])
    graph_changepoint = trackdependencies(model_changepoint)
    @testdependencies(model_changepoint, λ₁, λ₂, τ, y[1], y[2], y[3])
    @test_nothrow sample(model_changepoint, Gibbs(AutoConditional(:τ), MH(:λ₁, :λ₂)), 2)
end


###########################################################################
# @model function reverse_deps(x)
#     m = Vector{Float64}(undef, 2)
#     m[1] ~ Normal()
#     m[2] ~ Normal()
#     x ~ MvNormal(m)
# end

# model_reverse_deps = reverse_deps([0.1, -0.2])
# graph_reverse_deps = trackdependencies(model_reverse_deps)
# @testdependencies(model_reverse_deps, m[1], m[2], x)


##########################################################################
#########################################################################
### TEST TOGGLES

test_bernoulli()
test_gmm()
test_gmm_loopy()
test_gmm_shifted()
test_hmm()
# test_imm()
# test_changepoint()


