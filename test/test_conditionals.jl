include("models.jl")


function test_bernoulli()
    model_bernoulli = bernoulli_example()
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


function test_gmm()
    model_gmm = gmm_example()
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


function test_gmm_shifted()
    model_gmm_shifted = gmm_shifted([0.1, -0.05, 1.0], 2)
    graph_gmm_shifted = trackdependencies(model_gmm_shifted)
    @testdependencies(model_gmm_shifted, μ[1], μ[2], w, z[1], z[2], z[3], x[1], x[2], x[3])
    cond_gmm_shifted_z = StaticConditional(model_gmm_shifted, :z)
    @test_nothrow sample(model_gmm_shifted, Gibbs(cond_gmm_shifted_z, MH(:w, :μ)), 500)
    @test_nothrow sample(model_gmm_shifted, Gibbs(cond_gmm_shifted_z, HMC(0.01, 10, :w, :μ)), 500)
end


function test_hmm()
    model_hmm = hmm_example()
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


function test_imm_stick()
    model_imm_stick = imm_stick_example()
    graph_imm_stick = trackdependencies(model_imm_stick)
    
    # we leave out the μs, because there might be 1--3 of them
    @testdependencies(model_imm_stick,
                      v, # 10 - 1 sticks
                      μ, # 10 cluster centers
                      # z, # 9 data points
                      y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9])
    cond_imm_stick = StaticConditional(model_imm_stick, :z)
    @test_nothrow sample(model_imm_stick, Gibbs(cond_imm_stick, MH(:μ, :v)), 500)
    @test_nothrow sample(model_imm_stick, Gibbs(cond_imm_stick, HMC(0.01, 10, :μ, :v)), 500)

    # Analytic tests
    y = graph_imm_stick[2].value
    α = graph_imm_stick[4].value
    K = graph_imm_stick[6].value
    N = graph_imm_stick[7].value
    v = graph_imm_stick[12].value
    # z = graph_imm_stick[16].value
    z = [v.value for v in values(graph_imm_stick.statements)
         if v isa AutoGibbs.Assumption && DynamicPPL.subsumes(@varname(z), v.vn)]
    # μ = graph_imm_stick[19].value
    μ = (v.value for v in values(graph_imm_stick.statements)
         if v isa AutoGibbs.Assumption && DynamicPPL.subsumes(@varname(μ), v.vn)) |> first


    D_w = Categorical(stickbreak(v))
    analytic_conditionals = map(1:N) do n
        p̃ = [exp(logpdf(D_w, z) + logpdf(Normal(μ[z], 1.0), y[n])) for z in support(D_w)]
        @varname(z[n]) => Categorical(p̃ ./ sum(p̃))
    end
    # analytic_conditional = Product([D for (vn, D) in analytic_conditionals])
    # @info "stick-breaking IMM analytic conditionals" analytic_conditionals
    
    θ = AutoGibbs.sampled_values(graph_imm_stick)
    
    local calculated_conditionals
    @test_nothrow calculated_conditionals = conditionals(graph_imm_stick, @varname(z))
    # @info "stick-breaking IMM calculated conditionals" Dict(
        # vn => cond(θ) for (vn, cond) in calculated_conditionals)

    for (vn, analytic_conditional) in analytic_conditionals
        # @show vn => probs(calculated_conditionals[vn]), probs(analytic_conditional)
        @test issimilar(calculated_conditionals[vn](θ), analytic_conditional)
    end
end


function test_changepoint()
    model_changepoint = changepoint([1.1, 0.9, 0.2])
    graph_changepoint = trackdependencies(model_changepoint)
    @testdependencies(model_changepoint, λ₁, λ₂, τ, y[1], y[2], y[3])
    @test_nothrow sample(model_changepoint, Gibbs(AutoConditional(:τ), MH(:λ₁, :λ₂)), 20)
end


# function test_reverseps()
#     model_reverse_deps = reverse_deps([0.1, -0.2])
#     graph_reverse_deps = trackdependencies(model_reverse_deps)
#     @testdependencies(model_reverse_deps, m[1], m[2], x)
# end


##########################################################################
#########################################################################
### TEST TOGGLES

test_bernoulli()
test_gmm()
test_gmm_loopy()
test_gmm_shifted()
test_hmm()
test_imm_stick()
test_changepoint()


