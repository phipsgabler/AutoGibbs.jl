mutable struct CRP <: Distributions.DiscreteMultivariateDistribution
    N::Int
    α::Float64
end

CRP(N, α, G₀) = CRP(N, α)
Base.show(io::IO, d::CRP) = print(io, "CRP(", d.N, ", ", d.α, ")")

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

# `Product`s can be treated as an array of iid variables
function (c::GibbsConditional{V, L})(θ) where {
    V<:VarName, L<:LogLikelihood{<:CRP}}

    N = c.base.dist.N
    Ωs = [1:n for n = 1:N]
    conditionals = similar(Ωs, DiscreteNonParametric)
    
    for index in eachindex(Ωs, conditionals)
        sub_vn = DynamicPPL.VarName(c.vn, (DynamicPPL.getindexing(c.vn)..., (index,)))
        θs_on_support = fixvalues(θ, sub_vn => Ωs[index])
        logtable = map(θs_on_support) do θ′
            c.base(θ′) + reduce(+,
                                (_likelihood(vn, β, c, θ′) for (vn, β) in c.blanket if vn == sub_vn),
                                init=0.0)
        end
        conditionals[index] = DiscreteNonParametric(Ωs[index], softmax!(vec(logtable)))
    end

    return Product(conditionals)
end


function _likelihood(vn, β, c, θ)
    try
        return β(θ)
    catch
        G₀ = c.base.args[3](θ)
        return logpdf(G₀, θ[vn])
    end
end


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
