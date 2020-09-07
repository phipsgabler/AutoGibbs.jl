
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
    z = [graph_imm[20].value, graph_imm[37].value, graph_imm[53].value]
    μ = [v.value for v in values(graph_imm.statements)
         if v isa AutoGibbs.Assumption && DynamicPPL.subsumes(@varname(μ), v.vn)]
    y = graph_imm[2].value
    K = graph_imm[56].value
    N = graph_imm[7].value


    CRP(h) = ChineseRestaurantProcess(DirichletProcess(1.0), h)

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
    analytic_conditionals_coll = [@varname(z[1]) => Categorical(p_z1_coll ./ sum(p_z1_coll)),
                                  @varname(z[2]) => Categorical(p_z2_coll ./ sum(p_z2_coll)),
                                  @varname(z[3]) => Categorical(p_z3_coll ./ sum(p_z3_coll))]
    @info "IMM analytic, collapsed" analytic_conditionals

    
    θ = AutoGibbs.sampled_values(graph_imm)
    
    local calculated_conditionals
    @test_nothrow calculated_conditionals = conditionals(graph_imm, @varname(z))
    @info "IMM calculated" Dict(vn => cond(θ) for (vn, cond) in calculated_conditionals)
    
    # for (vn, analytic_conditional) in analytic_conditionals
    #     # @show vn => probs(calculated_conditionals[vn]), probs(analytic_conditional)
    #     @test issimilar(calculated_conditionals[vn](θ), analytic_conditional)
    # end
end


###########################################################################
mutable struct CRP <: Distributions.DiscreteMultivariateDistribution
    N::Int
    α::Float64
    nk::Vector{Int}
    K::Int
    
    CRP(N, α, G₀) = new(N, α)
end

Base.show(io::IO, d::CRP) = print(io, "CRP(", d.N, ", ", d.α, ")")


function Distributions.rand(rng::AbstractRNG, d::CRP)
    d.nk = Int[]
    d.K = 0
    z = Vector{Int}(undef, d.N)

    D = ChineseRestaurantProcess(DirichletProcess(d.α), d.nk)
    
    for n = 1:d.N
        z[n] = rand(rng, D)
        d.nk = update_histogram!(d.nk, z[n])
        d.K = max(d.K, z[n])
    end

    return z
end

function Distributions.logpdf(d::CRP, z::AbstractVector{<:Int})
    D = ChineseRestaurantProcess(DirichletProcess(d.α), d.nk)
    return mapreduce(zi -> logpdf(D, zi), +, z; init=0.0)
end

@model function imm_oneshot(y, α, ::Type{T}=Vector{Float64}) where {T}
    N = length(y)

    G₀ = Normal()
    z ~ CRP(N, α, G₀)
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
    # data from R. Neal paper: [-1.48, -1.40, -1.16, -1.08, -1.02, 0.14, 0.51, 0.53, 0.78]
    model_imm_oneshot = imm_oneshot([-1.02, 0.14, 0.51], 10.0)
    graph_imm_oneshot = trackdependencies(model_imm_oneshot)

    calculated_conditionals = conditionals(graph_imm_oneshot, @varname(z))
end


###########################################################################
@model function imm_manual(y, α, ::Type{T}=Vector{Float64}) where {T}
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
    # data from R. Neal paper: [-1.48, -1.40, -1.16, -1.08, -1.02, 0.14, 0.51, 0.53, 0.78]
    model_imm_manual = imm_manual([-1.02, 0.14, 0.51], 10.0)
    graph_imm_manual = trackdependencies(model_imm_manual)

    calculated_conditionals = conditionals(graph_imm_manual, @varname(z))

        
    # # 𝓅(zₙ = k| z₁, ..., zₙ₋₁, μ, yₙ) ∝ (∏_{i = z ≥ n} 𝓅(zᵢ | z₁,...,zᵢ)) 𝓅(yₙ | zₙ, μ)
    # function cond(n, k)
    #     # 𝓅(zₙ = k | z₁, ..., zₙ₋₁)
    #     l = logpdf(CRP(z[1:n-1]), k)

    #     # 𝓅(zₙ₊ᵢ | z₁, ..., zₙ = k, ..., zₙ₊ᵢ₋₁) for i = n+1 .. N
    #     for i = n+1:N
    #         l += logpdf(CRP([j == n ? k : z[j] for j = 1:i-1]), z[i])
    #     end

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
end



##########################################################################
#########################################################################
### TEST TOGGLES
test_imm()
test_imm_manual()
test_imm_oneshot()
