# data from R. Neal paper: 
const data_neal = [-1.48, -1.40, -1.16, -1.08, -1.02, 0.14, 0.51, 0.53, 0.78]
const α_neal = 10.0


function update_histogram!(histogram, bin)
    if bin > length(histogram)
        push!(histogram, 1)
    else
        histogram[bin] += 1
    end

    return histogram
end

DP(α, G₀) = DirichletProcess(α)


###########################################################################

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
@model function imm_stick(y, α, K)
    N = length(y)

    μ ~ filldist(Normal(0), K)

    crm = DirichletProcess(α)
    v ~ filldist(StickBreakingProcess(crm), K - 1)
    w = stickbreak(v)

    z = Vector{Int}(undef, N)
    for n = 1:N
        z[n] ~ Categorical(w)
        y[n] ~ Normal(μ[z[n]], 1.0)
    end
end

function test_imm_stick()
    model_imm_stick = imm_stick(data_neal, α_neal, 10)
    graph_imm_stick = trackdependencies(model_imm_stick)
    
    # we leave out the μs, because there might be 1--3 of them
    @testdependencies(model_imm_stick,
                      v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], # 10 - 1 sticks
                      μ[1], μ[2], μ[3], μ[4], μ[5], μ[6], μ[7], μ[8], μ[9], μ[10], # 10 cluster centers
                      z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9], # 9 data points
                      y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9])
    cond_imm_stick = StaticConditional(model_imm_stick, :z)
    @test_nothrow sample(model_imm_stick, Gibbs(cond_imm_stick, MH(:μ, :v)), 500)
    @test_nothrow sample(model_imm_stick, Gibbs(cond_imm_stick, HMC(0.01, 10, :μ, :v)), 500)

    # Analytic tests
    μ = graph_imm_stick[9].value
    v = graph_imm_stick[14].value
    z = [v.value for v in values(graph_imm_stick.statements)
         if v isa AutoGibbs.Assumption && DynamicPPL.subsumes(@varname(z), v.vn)]
    y = graph_imm_stick[2].value
    K = graph_imm_stick[6].value
    N = graph_imm_stick[7].value
    α = graph_imm_stick[4].value

    D_w = Categorical(stickbreak(v))
    p_z = [logpdf(D_w, z) + logpdf(Normal(μ[z]), y[n]) for z = 1:K, n = 1:N]
    analytic_conditionals = map(1:N) do n
        p̃ = [logpdf(D_w, z) + logpdf(Normal(μ[z]), y[n]) for z = 1:K]
        @varname(z[n]) => Categorical(p̃ ./ sum(p̃))
    end
    @info "stick-breaking IMM analytic conditionals" analytic_conditionals
    
    θ = AutoGibbs.sampled_values(graph_imm_stick)
    
    local calculated_conditionals
    @test_nothrow calculated_conditionals = conditionals(graph_imm_stick, @varname(z))
    @info "stick-breaking IMM calculated conditionals" Dict(
        vn => cond(θ) for (vn, cond) in calculated_conditionals)
    
    for (vn, analytic_conditional) in analytic_conditionals
        # @show vn => probs(calculated_conditionals[vn]), probs(analytic_conditional)
        @test issimilar(calculated_conditionals[vn](θ), analytic_conditional)
    end
end


###########################################################################
# version as this would usually be written in Turing, with the slight
# modification that we need to associate G₀ with z (to calculate marginals of
# y for "unobserved clusters")
@model function imm(y, α, ::Type{T}=Array{Float64, 1}) where {T}
    N = length(y)

    K = 0
    nk = Vector{Int}()
    z = Vector{Int}(undef, N)

    G₀ = Normal()
    
    for n = 1:N
        z[n] ~ ChineseRestaurantProcess(DP(α, G₀), nk)
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
    model_imm = imm(data_neal[5:7], α_neal)
    graph_imm = trackdependencies(model_imm)
    
    # we leave out the μs, because there might be 1--3 of them
    @testdependencies(model_imm, z[1], z[2], z[3], y[1], y[2], y[3])
    # @test_nothrow sample(model_imm, Gibbs(AutoConditional(:z), HMC(0.01, 10, :μ)), 2)

    # for comparison:
    # sample(model_imm, Gibbs(MH(:z => filldist(Categorical(9), 9)), HMC(0.01, 10, :μ)), 2)


    # Analytic tests
    z = [graph_imm[20].value, graph_imm[37].value, graph_imm[53].value]
    μ = [v.value for v in values(graph_imm.statements)
         if v isa AutoGibbs.Assumption && DynamicPPL.subsumes(@varname(μ), v.vn)]
    y = graph_imm[2].value
    K = graph_imm[56].value
    N = graph_imm[7].value


    CRP(h) = ChineseRestaurantProcess(DirichletProcess(α_neal), h)

    # exploiting exchangeability:
    # 𝓅(zₙ | z₋ₙ, μ, yₙ) ∝ 𝓅(zₙ | z₋ₙ) 𝓅(yₙ | zₙ, μ)
    function cond_collapsed(n, k)
        # nk is the histogram of z without z[k]
        nk = zeros(Int, N)
        for i in eachindex(z)
            i != k && (nk[z[i]] += 1)
        end
        
        l = logpdf(CRP(nk), z[k])
        
        if k <= K
            # 𝓅(yₙ | zₙ = k, μ)
            l += logpdf(Normal(μ[k]), y[n])
        else # k == K + 1
            # 𝓅(yₙ | zₙ = K + 1, μ) = ∫ 𝓅(yₙ | m) 𝓅(m) dm = pdf(G₀, yₙ)
            l += logpdf(Normal(), y[n])
        end

        return exp(l)
    end
    
    p_z1_coll = cond_collapsed.(1, 1:1)
    p_z2_coll = cond_collapsed.(2, 1:2)
    p_z3_coll = cond_collapsed.(3, 1:3)
    analytic_conditionals = [@varname(z[1]) => Categorical(p_z1_coll ./ sum(p_z1_coll)),
                             @varname(z[2]) => Categorical(p_z2_coll ./ sum(p_z2_coll)),
                             @varname(z[3]) => Categorical(p_z3_coll ./ sum(p_z3_coll))]
    @info "normal IMM analytic conditionals (collapsed)" analytic_conditionals

    
    θ = AutoGibbs.sampled_values(graph_imm)
    
    local calculated_conditionals
    @test_nothrow calculated_conditionals = conditionals(graph_imm, @varname(z))
    @info "normal IMM calculated conditionals" Dict(vn => cond(θ) for (vn, cond) in calculated_conditionals)
    
    for (vn, analytic_conditional) in analytic_conditionals
        # @show vn => probs(calculated_conditionals[vn]), probs(analytic_conditional)
        @test issimilar(calculated_conditionals[vn](θ), analytic_conditional)
    end
end


###########################################################################
@model function imm_oneshot(y, α, ::Type{T}=Array{Float64, 1}) where {T}
    N = length(y)
    G₀ = Normal()
    z ~ AutoGibbs.CRP(N, α, G₀)
    K = maximum(z)

    μ = T(undef, K)
    for k = 1:K
        μ[k] ~ G₀
    end
    
    for n = 1:N
        y[n] ~ Normal(μ[z[n]], 1.0)
    end
end

function test_imm_oneshot()
    model_imm_oneshot = imm_oneshot(data_neal[5:7], α_neal)
    graph_imm_oneshot = trackdependencies(model_imm_oneshot)
    
    # we leave out the μs, because there might be 1--3 of them
    @testdependencies(model_imm_oneshot, z, y[1], y[2], y[3])
    # @test_nothrow sample(model_imm_oneshot, Gibbs(AutoConditional(:z), HMC(0.01, 10, :μ)), 2)

    # for comparison:
    # sample(model_imm_oneshot, Gibbs(MH(:z => filldist(Categorical(9), 9)), HMC(0.01, 10, :μ)), 2)

    # Analytic tests
    z = graph_imm_oneshot[10].value
    μ = [v.value for v in values(graph_imm_oneshot.statements)
         if v isa AutoGibbs.Assumption && DynamicPPL.subsumes(@varname(μ), v.vn)]
    y = graph_imm_oneshot[2].value
    K = graph_imm_oneshot[11].value
    N = graph_imm_oneshot[7].value

    CRP(h) = ChineseRestaurantProcess(DirichletProcess(α_neal), h)

    # exploiting exchangeability:
    # 𝓅(zₙ | z₋ₙ, μ, yₙ) ∝ 𝓅(zₙ | z₋ₙ) 𝓅(yₙ | zₙ, μ)
    function cond_collapsed(n, k)
        # nk is the histogram of z without z[k]
        nk = zeros(Int, N)
        for i in eachindex(z)
            i != k && (nk[z[i]] += 1)
        end
        
        l = logpdf(CRP(nk), z[k])
        
        if k <= K
            # 𝓅(yₙ | zₙ = k, μ)
            l += logpdf(Normal(μ[k]), y[n])
        else # k == K + 1
            # 𝓅(yₙ | zₙ = K + 1, μ) = ∫ 𝓅(yₙ | m) 𝓅(m) dm = pdf(G₀, yₙ)
            l += logpdf(Normal(), y[n])
        end

        return exp(l)
    end
    
    p_z1_coll = cond_collapsed.(1, 1:1)
    p_z2_coll = cond_collapsed.(2, 1:2)
    p_z3_coll = cond_collapsed.(3, 1:3)
    analytic_conditionals = [@varname(z[1]) => Categorical(p_z1_coll ./ sum(p_z1_coll)),
                             @varname(z[2]) => Categorical(p_z2_coll ./ sum(p_z2_coll)),
                             @varname(z[3]) => Categorical(p_z3_coll ./ sum(p_z3_coll))]
    @info "normal IMM analytic conditionals (collapsed)" analytic_conditionals
    
    # θ = AutoGibbs.sampled_values(graph_imm_oneshot)
    
    # local calculated_conditionals
    # @test_nothrow calculated_conditionals = conditionals(graph_imm_oneshot, @varname(z))
    # @info "one-shot IMM calculated conditionals" Dict(vn => cond(θ) for (vn, cond) in calculated_conditionals)
    
    # for (vn, analytic_conditional) in analytic_conditionals
    #     # @show vn => probs(calculated_conditionals[vn]), probs(analytic_conditional)
    #     @test issimilar(calculated_conditionals[vn](θ), analytic_conditional)
    # end
end


###########################################################################
@model function imm_manual(y, α, ::Type{T}=Array{Float64, 1}) where {T}
    N = length(y)

    K = 0
    nk = Vector{Int}()
    z = Vector{Int}(undef, N)

    G₀ = Normal()
    
    for n = 1:N
        probs = [nk; α] ./ (n + α - 1)
        z[n] ~ Categorical(probs)
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

function test_imm_manual()
    model_imm_manual = imm_manual(data_neal[5:7], α_neal)
    graph_imm_manual = trackdependencies(model_imm_manual)

    @testdependencies(model_imm_manual, z[1], z[2], z[3], y[1], y[2], y[3])
    # @test_nothrow sample(model_imm_manual, Gibbs(AutoConditional(:z), HMC(0.01, 10, :μ)), 2)

    calculated_conditionals = conditionals(graph_imm_manual, @varname(z))

    # # log 𝓅(z₁, ..., zₙ)
    # function logpdf_crp(z)
    #     N = length(z)
    #     l = 0.0
    #     nk = Int[]
    #     K = 0
    #     log_α = log(α)
        
    #     for n = 1:N
    #         if z[n] <= K
    #             l += log(z[n]) - log(n + α - 1)
    #             nk[z[n]] += 1
    #         else
    #             l += log_α - log(n + α - 1)
    #             push!(nk, 1)
    #             K += 1
    #         end
            
    #     end
    #     return l
    # end
    
    # # 𝓅(zₙ = k| z₁, ..., zₙ₋₁, μ, yₙ) ∝ (∏_{i = z ≥ n} 𝓅(zᵢ | z₁,...,zᵢ)) 𝓅(yₙ | zₙ, μ)
    # function cond(n, k)
    #     # 𝓅(zₙ = k | z₁, ..., zₙ₋₁)
    #     l = logpdf_crp([j == n ? k : z[j] for j = 1:n])

    #     if k <= K
    #         # 𝓅(yₙ | zₙ = k, μ)
    #         l += logpdf(Normal(μ[k]), y[n])
    #     else
    #         # 𝓅(yₙ | zₙ = K + 1, μ) = ∫ 𝓅(yₙ | m) 𝓅(m) dm = pdf(G₀, yₙ)
    #         l += logpdf(Normal(), y[n])
    #     end

    #     return exp(l)
    # end

    # p_z1 = cond.(1, 1:1)
    # p_z2 = cond.(2, 1:2)
    # p_z3 = cond.(3, 1:3)
    # analytic_conditionals = [@varname(z[1]) => Categorical(p_z1 ./ sum(p_z1)),
    #                          @varname(z[2]) => Categorical(p_z2 ./ sum(p_z2)),
    #                          @varname(z[3]) => Categorical(p_z3 ./ sum(p_z3))]
    # @info "IMM analytic" analytic_conditionals

    θ = AutoGibbs.sampled_values(graph_imm_manual)

    local calculated_conditionals
    @test_nothrow calculated_conditionals = conditionals(graph_imm_manual, @varname(z))
    @info "manual IMM calculated conditionals" Dict(vn => cond(θ) for (vn, cond) in calculated_conditionals)
end



##########################################################################
#########################################################################
### TEST TOGGLES
test_imm_stick()
# test_imm()
# test_imm_oneshot()
# test_imm_manual()
