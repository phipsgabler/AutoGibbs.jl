# @model function bernoulli_mixture(x)
#     # Mixture prior.
#     w ~ Dirichlet(2, 1.0)

#     # Latent probability.
#     p ~ DiscreteNonParametric([0.3, 0.7], w)

#     # Observation.
#     x ~ Bernoulli(p)
# end

# model_bernoulli = bernoulli_mixture(false)
# graph_bernoulli = trackdependencies(model_bernoulli)
# @testdependencies(model_bernoulli, w, p, x)
# @test_nothrow sample(model_bernoulli, Gibbs(AutoConditional(:p), MH(:w)), 2)


# let w = graph_bernoulli[4].value,
#     p = graph_bernoulli[6].value,
#     x = graph_bernoulli[2].value,
#     p_1 = w[1] * pdf(Bernoulli(0.3), x),
#     p_2 = w[2] * pdf(Bernoulli(0.7), x),
#     Z = p_1 + p_2

#     # 𝓅(p | w, x) ∝ 𝓅(p | w) * 𝓅(x | p)
#     analytic_conditional = DiscreteNonParametric([0.3, 0.7], [p_1 / Z, p_2 / Z])
#     @info "Bernoulli analytic" analytic_conditional
#     θ = AutoGibbs.sampled_values(graph_bernoulli)
    
#     local calculated_conditional
#     @test_nothrow calculated_conditional = conditionals(graph_bernoulli, @varname(p))[@varname(p)]
#     @info "Bernoulli calculated" calculated_conditional(θ)
    
#     @test issimilar(calculated_conditional(θ), analytic_conditional)
# end


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

model_gmm = gmm([0.1, -0.05, 1.0], 2)
graph_gmm = trackdependencies(model_gmm)
@testdependencies(model_gmm, μ, w, z, x[1], x[2], x[3])
@test_nothrow sample(model_gmm, Gibbs(AutoConditional(:z), MH(:w, :μ)), 2)
@test_nothrow sample(model_gmm, Gibbs(AutoConditional(:z), HMC(0.01, 10, :w, :μ)), 2)

let μ = graph_gmm[7].value,
    w = graph_gmm[9].value,
    z = graph_gmm[13].value,
    x = graph_gmm[2].value,
    p_1 = w[1] .* pdf.(Normal(μ[1], 1.0), x),
    p_2 = w[2] .* pdf.(Normal(μ[2], 1.0), x),
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


# ###########################################################################
# @model function gmm_loopy(x, K, ::Type{T}=Float64) where {T<:Real}
#     N = length(x)

#     # Cluster centers.
#     μ = Vector{T}(undef, K)
#     for k = 1:K
#         μ[k] ~ Normal()
#     end

#     # Cluster association prior.
#     w ~ Dirichlet(K, 1.0)

#     # Cluster assignments & observations.
#     z = Vector{Int}(undef, N)
#     for n = 1:N
#         z[n] ~ DiscreteNonParametric(1:K, w)
#         x[n] ~ Normal(μ[z[n]], 1.0)
#     end
# end

# model_gmm_loopy = gmm_loopy([0.1, -0.05, 1.0], 2)
# graph_gmm_loopy = trackdependencies(model_gmm_loopy)
# @testdependencies(model_gmm_loopy, μ[1], μ[2], w, z[1], z[2], z[3], x[1], x[2], x[3])
# @test_nothrow sample(model_gmm_loopy, Gibbs(AutoConditional(:z), MH(:w, :μ)), 2)
# @test_nothrow sample(model_gmm_loopy, Gibbs(AutoConditional(:z), HMC(0.01, 10, :w, :μ)), 2)

# let μ = [graph_gmm_loopy[19].value, graph_gmm_loopy[28].value],
#     w = graph_gmm_loopy[31].value,
#     z = [graph_gmm_loopy[43].value, graph_gmm_loopy[62].value, graph_gmm_loopy[80].value],
#     x = graph_gmm_loopy[2].value,
#     p_1 = w[1] .* pdf.(Normal(μ[1], 1.0), x),
#     p_2 = w[2] .* pdf.(Normal(μ[2], 1.0), x),
#     (Z1, Z2, Z3) = p_1 .+ p_2

#     # 𝓅(zᵢ | μ, w, x, z₋ᵢ) ∝ 𝓅(zᵢ | w) * 𝓅(xᵢ | zᵢ, μ)
#     analytic_conditionals = [@varname(z[1]) => Categorical([p_1[1], p_2[1]] ./ Z1),
#                              @varname(z[2]) => Categorical([p_1[2], p_2[2]] ./ Z2),
#                              @varname(z[3]) => Categorical([p_1[3], p_2[3]] ./ Z3)]
#     @info "Loopy GMM analytic" analytic_conditionals
#     θ = AutoGibbs.sampled_values(graph_gmm_loopy)
    
#     local calculated_conditionals
#     @test_nothrow calculated_conditionals = conditionals(graph_gmm_loopy, @varname(z))
#     @info "Loopy GMM calculated" Dict(vn => cond(θ) for (vn, cond) in calculated_conditionals)
    
#     for (vn, analytic_conditional) in analytic_conditionals
#         @test issimilar(calculated_conditionals[vn](θ), analytic_conditional)
#     end
# end 


# ###########################################################################
# # same as gmm_loopy, but with an affine transformation on μ.
# @model function gmm_shifted(x, K, ::Type{T}=Float64) where {T<:Real}
#     N = length(x)

#     # Cluster centers.
#     μ = Vector{T}(undef, K)
#     for k = 1:K
#         μ[k] ~ Normal()
#     end

#     # Cluster association prior.
#     w ~ Dirichlet(K, 1.0)

#     # Cluster assignments & observations.
#     z = Vector{Int}(undef, N)
#     for n = 1:N
#         z[n] ~ DiscreteNonParametric(1:K, w)
#         x[n] ~ Normal(4μ[z[n]] - 1, 1.0)
#     end
# end

# model_gmm_shifted = gmm_shifted([0.1, -0.05, 1.0], 2)
# graph_gmm_shifted = trackdependencies(model_gmm_shifted)
# @testdependencies(model_gmm_shifted, μ[1], μ[2], w, z[1], z[2], z[3], x[1], x[2], x[3])
# @test_nothrow sample(model_gmm_shifted, Gibbs(AutoConditional(:z), MH(:w, :μ)), 2)
# @test_nothrow sample(model_gmm_shifted, Gibbs(AutoConditional(:z), HMC(0.01, 10, :w, :μ)), 2)


# ###########################################################################
# # K clusters, each one around i for i = 1:K with variance 0.5
# @model function hmm(x, K, ::Type{T}=Float64) where {T<:Real}
#     N = length(x)

#     # State sequence.
#     s = zeros(Int, N)

#     # Emission matrix.
#     m = Vector{T}(undef, K)

#     # Transition matrix.
#     T = Vector{Vector{T}}(undef, K)

#     # Assign distributions to each element
#     # of the transition matrix and the
#     # emission matrix.
#     for i = 1:K
#         T[i] ~ Dirichlet(K, 1.0)
#         m[i] ~ Normal(i, 0.5)
#     end
    
#     # Observe each point of the input.
#     # note that `Categorical(K)` does not work, because it is an alias method!
#     s[1] ~ DiscreteNonParametric(1:K, fill(1/K, K))
#     x[1] ~ Normal(m[s[1]], 0.1)

#     for i = 2:N
#         s[i] ~ DiscreteNonParametric(1:K, T[s[i-1]])
#         x[i] ~ Normal(m[s[i]], 0.1)
#     end
# end

# model_hmm = hmm([1.1, 0.95, 2.2], 2)
# graph_hmm = trackdependencies(model_hmm)
# @testdependencies(model_hmm, T[1], T[2], m[1], m[2], s[1], s[2], s[3], x[1], x[2], x[3])
# @test_nothrow sample(model_hmm, Gibbs(AutoConditional(:s), MH(:m, :T)), 2)
# @test_nothrow sample(model_hmm, Gibbs(AutoConditional(:s), HMC(0.01, 10, :m, :T)), 2)

# let T = [graph_hmm[23].value, graph_hmm[38].value],
#     m = [graph_hmm[29].value, graph_hmm[44].value],
#     s1 = graph_hmm[51].value,
#     s2 = graph_hmm[72].value,
#     s3 = graph_hmm[93].value,
#     x = graph_hmm[2].value,
#     D_obs_1 = Normal(m[1], 0.1),
#     D_obs_2 = Normal(m[2], 0.1),
#     p_s1_1 = pdf(Categorical(2), 1) * pdf(Categorical(T[1]), s2) * pdf(D_obs_1, x[1]),
#     p_s1_2 = pdf(Categorical(2), 2) * pdf(Categorical(T[2]), s2) * pdf(D_obs_2, x[1]),
#     p_s2_1 = pdf(Categorical(T[s1]), 1) * pdf(Categorical(T[1]), s3) * pdf(D_obs_1, x[2]),
#     p_s2_2 = pdf(Categorical(T[s1]), 2) * pdf(Categorical(T[2]), s3) * pdf(D_obs_2, x[2]),
#     p_s3_1 = pdf(Categorical(T[s2]), 1) * pdf(D_obs_1, x[3]),
#     p_s3_2 = pdf(Categorical(T[s2]), 2) * pdf(D_obs_2, x[3]),
#     Z_1 = p_s1_1 + p_s1_2,
#     Z_2 = p_s2_1 + p_s2_2,
#     Z_3 = p_s3_1 + p_s3_2

#     # 𝓅(s₁ | T, m, s₋₁, x) ∝ 𝓅(s₁) 𝓅(s₂ | s₁, T) 𝓅(x₁ | s₁, m)
#     #  𝓅(sᵢ | T, m, s₋ᵢ, x) ∝  𝓅(sᵢ | sᵢ₋₁, T) 𝓅(sᵢ₊₁ | sᵢ, T) 𝓅(xᵢ | sᵢ, m) (for i ≥ 2)
#     analytic_conditionals = [@varname(s[1]) => Categorical([p_s1_1, p_s1_2] ./ Z_1),
#                              @varname(s[2]) => Categorical([p_s2_1, p_s2_2] ./ Z_2),
#                              @varname(s[3]) => Categorical([p_s3_1, p_s3_2] ./ Z_3)]
#     θ = AutoGibbs.sampled_values(graph_hmm)
#     @info "HMM analytic" analytic_conditionals
    
#     local calculated_conditionals
#     @test_nothrow calculated_conditionals = conditionals(graph_hmm, @varname(s))
#     @info "HMM calculated" Dict(vn => cond(θ) for (vn, cond) in calculated_conditionals)
    
#     for (vn, analytic_conditional) in analytic_conditionals
#         # @show vn => probs(calculated_conditionals[vn]), probs(analytic_conditional)
#         @test issimilar(calculated_conditionals[vn](θ), analytic_conditional)
#     end
# end


###########################################################################
# update_histogram!(nk, bin) = (nk[bin] += 1; nk)

# @model function imm(x)
#     N = length(x)

#     nk = zeros(Int, N)
#     K = 0
#     z = zeros(Int, N)
    
#     for n = 1:N
#         z[n] ~ ChineseRestaurantProcess(DirichletProcess(1.0), nk)
#         nk = update_histogram!(nk, z[n])
#         K = max(K, z[n])
#     end

#     μ ~ filldist(Normal(), K)

#     for n = 1:N
#         x[n] ~ Normal(μ[z[n]], 1.0)
#     end
# end

# model_imm = imm([0.1, -0.05, 1.0])
# graph_imm = trackdependencies(model_imm)
# @testdependencies(model_imm, z[1], z[2], z[3], μ, x[1], x[2], x[3])
# @test_nothrow sample(model_imm, Gibbs(AutoConditional(:z), MH(:μ)), 2)
# @test_nothrow sample(model_imm, Gibbs(AutoConditional(:z), HMC(0.01, 10, :μ)), 2)


###########################################################################
# @model function changepoint(y)
#     N = length(y)
#     α = 1 / mean(y)
#     λ₁ ~ Exponential(α)
#     λ₂ ~ Exponential(α)
#     τ ~ DiscreteUniform(1, N)
    
#     for n in 1:N
#         y[n] ~ Poisson(τ > N ? λ₁ : λ₂)
#     end
# end

# model_changepoint = changepoint([1.1, 0.9, 0.2])
# graph_changepoint = trackdependencies(model_changepoint)
# @testdependencies(model_changepoint, λ₁, λ₂, τ, y[1], y[2], y[3])
# @test_nothrow sample(model_changepoint, Gibbs(AutoConditional(:τ), MH(:λ₁, :λ₂)), 2)


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
