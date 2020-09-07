using Turing
using Turing.RandomMeasures
using Distributions
using Random
using Libtask
using AutoGibbs


const data = [-1.48, -1.40, -1.16, -1.08, -1.02, 0.14, 0.51, 0.53, 0.78]


function update_histogram!(histogram, bin)
    if bin > length(histogram)
        push!(histogram, 1)
    else
        histogram[bin] += 1
    end

    return histogram
end

function histogram(d, z)
    nk = Int[]
    K = 0

    for n in 1:d.N
        nk = update_histogram!(nk, z[n])
        K = max(K, z[n])
    end

    return nk, K
end


mutable struct CRP <: Distributions.DiscreteMultivariateDistribution
    N::Int
    α::Float64
end

CRP(N, α, G₀) = CRP(N, α)

function Distributions.rand(rng::AbstractRNG, d::CRP)
    nk = Int[]
    D = ChineseRestaurantProcess(DirichletProcess(d.α), nk)
    z = rand(rng, D, d.N)
end

function Distributions.logpdf(d::CRP, z::AbstractVector{<:Int})
    nk, K = histogram(d, z)
    D = ChineseRestaurantProcess(DirichletProcess(d.α), nk)
    return mapreduce(zi -> logpdf(D, zi), +, z; init=0.0)
end


function (c::GibbsConditional{V, L})(θ) where {
    V<:VarName, L<:LogLikelihood{<:CRP}}

    Ω = c.base.dist.N

    conditionals = similar(Ωs, DiscreteNonParametric)
    
    for index in eachindex(Ωs, independent_distributions, conditionals)
        sub_vn = DynamicPPL.VarName(c.vn, (DynamicPPL.getindexing(c.vn)..., (index,)))
        θs_on_support = fixvalues(θ, sub_vn => Ωs[index])
        logtable = map(θs_on_support) do θ′
            c.base(θ′) + reduce(+, (β(θ′) for (vn, β) in c.blanket if vn == sub_vn), init=0.0)
        end
        conditionals[index] = DiscreteNonParametric(Ωs[index], softmax!(vec(logtable)))
    end

    return Product(conditionals)
end


@model function imm_oneshot(y, α, ::Type{T}=Array{Float64, 1}) where {T}
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


model_imm_oneshot = imm_oneshot(data[5:7], 10.0)

sample(model_imm_oneshot, Gibbs(AutoConditional(:z), HMC(0.01, 10, :μ)), 2)
