using DataStructures: DefaultDict
using Distributions
using DynamicPPL


export conditional_dists


abstract type Cont end


abstract type ArgSpec <: Cont end

struct Fixed{T} <: ArgSpec
    value::T
end
Base.show(io::IO, arg::Fixed) = print(io, arg.value)
(arg::Fixed)(θ) = arg.value

struct Variable{TV<:VarName} <: ArgSpec
    vn::TV
end
Base.show(io::IO, arg::Variable) = print(io, "θ[", arg.vn, "]")
(arg::Variable)(θ) = getindex(θ, arg.vn)


struct Transformation{TF, N, TArgs<:NTuple{N, Cont}} <: Cont
    f::TF
    args::TArgs
end

function Base.show(io::IO, t::Transformation)
    print(io, t.f, "(")
    join(io, t.args, ", ")
    print(io, ")")
end

(t::Transformation)(θ) = t.f((arg(θ) for arg in t.args)...)


struct LogLikelihood{D<:Distribution, TVal, TArgs<:Tuple} <: Cont
    value::TVal
    args::TArgs
    
    function LogLikelihood(::Type{D}, value, args::NTuple{N, Cont}) where {D<:Distribution, N}
        return new{D, typeof(value), typeof(args)}(value, args)
    end
end

function Base.show(io::IO, ℓ::LogLikelihood{D}) where {D}
    print(io, "logpdf(", _shortname(D), "(")
    join(io, ℓ.args, ", ")
    print(io, "), ", ℓ.value, ")")
end

function (ℓ::LogLikelihood)(θ) where {D}
    return logpdf(D((arg(θ) for arg in ℓ.args)...), ℓ.value)
end


function continuations(graph)
    c = SortedDict{Reference, Cont}()
    
    function convertarg(arg)
        if arg isa Reference
            cont = c[arg]
            stmt = graph[arg]
            if cont isa LogLikelihood && !isnothing(stmt.vn)
                Variable(stmt.vn)
            elseif cont isa Transformation && !isnothing(stmt.definition)
                Variable(stmt.definition[1])
            else
                cont
            end
        else
            Fixed(arg)
        end
    end
    
    for (ref, stmt) in graph
        if stmt isa Union{Assumption, Observation}
            dist_call = stmt.dist

            if dist_call isa Call
                args = convertarg.(dist_call.args)
                D = typeof(getvalue(dist_call))
            elseif dist_call isa Constant
                dist = getvalue(dist_call)
                D = typeof(dist)
                args = Fixed.(params(dist))
            end

            if stmt isa Observation
                value = convertarg(getvalue(stmt))
            else
                value = Variable(stmt.vn)
            end
            c[ref] = LogLikelihood(D, value, args)
        # elseif stmt isa Call{<:Tuple, typeof(getindex)}
        #     vn, compound_ref = stmt.definition
        #     ix = getindexing(vn)[1]
        #     f, args = stmt.f, convertarg.(stmt.args)
        #     dist = graph[compound_ref].v[ix...]
        #     value = Variable(VarName(graph[compound_ref].vn, (ix,)))
        #     c[ref] = LogLikelihood(typeof(dist), value, )
        elseif stmt isa Call
            f, args = stmt.f, convertarg.(stmt.args)
            c[ref] = Transformation(f, args)
        elseif stmt isa Constant
            c[ref] = Fixed(getvalue(stmt))
        end
    end

    return c
end


Base.getindex(x::Union{Number, AbstractArray}, vn::VarName) = foldl((x, i) -> getindex(x, i...),
                                                                    DynamicPPL.getindexing(vn),
                                                                    init=x)

function conditionals(graph, varname)
    # There can be multiple tildes for one `varname`, e.g., `x[1], x[2]` both subsumed by `x`.
    # dists = Dict{VarName, Distribution}()
    # blankets = DefaultDict{Tuple{VarName, Union{Nothing, Tuple}}, Float64}(0.0)
    blankets = DefaultDict{VarName, Vector{Pair{Tuple, LogLikelihood}}}(
        Vector{Pair{Tuple, LogLikelihood}})
    conts = continuations(graph)
    
    for (ref, stmt) in graph
        if stmt isa Union{Assumption, Observation} && !isnothing(stmt.vn)
            vn, dist, value = stmt.vn, getvalue(stmt.dist), tovalue(graph, getvalue(stmt))

            # record distribution of this tilde if it matches the searched vn
            if DynamicPPL.subsumes(varname, vn)
                push!(blankets[vn], DynamicPPL.getindexing(vn) => conts[ref])
            end
            
            # add likelihood to all parents of which this RV is in the blanket
            for (p, ix) in parent_variables(graph, stmt)
                if any(DynamicPPL.subsumes(r, p.vn) for r in keys(blankets))
                    push!(blankets[p.vn], ix => conts[ref])
                    # @show stmt => (p, ix)
                    # ℓ = logpdf(dist, value)
                    # @show dist, value
                    # @show vn
                    # @show p.vn => ℓ
                    
                    # x, nothing ~> x; x, (1,) ~> x[1]
                    # @show (p.vn, ix) => ℓ
                    # blankets[(p.vn, ix)] += ℓ

                    # println("Found variable $vn being dependent on ($(p.vn), $ix) " *
                            # "with likelihood $ℓ and value $value")
                end
            end


        end
    end

    # result = Dict{VarName, Distribution}()
    
    # for (vn, d) in dists
    #     result[vn] = d
        
    #     for ((b, ix), ℓ) in blankets
    #         if DynamicPPL.subsumes(vn, b)
    #             # @show vn => (b, ix)
    #             if !isnothing(ix)
    #                 @assert DynamicPPL.subsumes(DynamicPPL.getindexing(vn), ix)
    #                 push!(result, vn => conditioned(d, ℓ, ix...))
    #                 # println("Update $vn from ($b, $ix) with $ℓ")
    #             else
    #                 push!(result, vn => conditioned(d, ℓ))
    #             end
    #         end
    #     end
    # end

    blankets
    
    # return result
end


# """
#     conditional_dists(graph, varname)

# Derive a dictionary of Gibbs conditionals for all assumption statements in `graph` that are subsumed
# by `varname`.
# """
# function conditional_dists(graph, varname)
#     # There can be multiple tildes for one `varname`, e.g., `x[1], x[2]` both subsumed by `x`.
#     dists = Dict{VarName, Distribution}()
#     blankets = DefaultDict{Tuple{VarName, Union{Nothing, Tuple}}, Float64}(0.0)
    
#     for (ref, stmt) in graph
#         if stmt isa Union{Assumption, Observation} && !isnothing(stmt.vn)
#             vn, dist, value = stmt.vn, getvalue(stmt.dist), tovalue(graph, getvalue(stmt))

#             # add likelihood to all parents of which this RV is in the blanket
#             for (p, ix) in parent_variables(graph, stmt)
#                 if any(DynamicPPL.subsumes(r, p.vn) for r in keys(dists))
#                     # @show stmt => (p, ix)
#                     ℓ = logpdf(dist, value)
#                     # @show dist, value
#                     # @show vn
#                     # @show p.vn => ℓ
                    
#                     # x, nothing ~> x; x, (1,) ~> x[1]
#                     # @show (p.vn, ix) => ℓ
#                     blankets[(p.vn, ix)] += ℓ

#                     # println("Found variable $vn being dependent on ($(p.vn), $ix) " *
#                             # "with likelihood $ℓ and value $value")
#                 end
#             end

#             # record distribution of this tilde if it matches the searched vn
#             if DynamicPPL.subsumes(varname, vn)
#                 dists[vn] = dist
#                 # @show vn => dist
#                 # println("Found variable $vn with value $value")
#             end
#         end
#     end

#     result = Dict{VarName, Distribution}()
    
#     for (vn, d) in dists
#         result[vn] = d
        
#         for ((b, ix), ℓ) in blankets
#             if DynamicPPL.subsumes(vn, b)
#                 # @show vn => (b, ix)
#                 if !isnothing(ix)
#                     @assert DynamicPPL.subsumes(DynamicPPL.getindexing(vn), ix)
#                     push!(result, vn => conditioned(d, ℓ, ix...))
#                     # println("Update $vn from ($b, $ix) with $ℓ")
#                 else
#                     push!(result, vn => conditioned(d, ℓ))
#                 end
#             end
#         end
#     end
    
#     return result
# end



DynamicPPL.getlogp(tilde::Union{Assumption, Observation}) = logpdf(tilde.dist, tilde.value)



"""
    conditioned(d0, blanket_logps)

Return an array of distributions for the RV with distribution `d0` within a Markov blanket.

Constructed as

    P[X = x | conditioned] ∝ P[X = x | parents(X)] * P[children(X) | parents(children(X))]

equivalent to

    logpdf(D, x) = logpdf(d0, x) + ∑ blanket_logps

where the factors `blanket_logps` are the log probabilities in the Markov blanket.

The result is an array to allow to condition `Product` distributions.
"""
function conditioned(d0::DiscreteUnivariateDistribution, blanket_logp::Real)
    local Ω

    try
        Ω = support(d0)
    catch
        throw(ArgumentError("Unable to get the support of $d0 (probably infinite)"))
    end
    
    logtable = logpdf.(d0, Ω) .+ blanket_logp
    # @show logpdf.(d0, Ω)
    # @show (softmax(logtable))
    return DiscreteNonParametric(Ω, softmax!(logtable))
end

conditioned(d0::DiscreteUnivariateDistribution, blanket_logp, ix) = conditioned(d0, blanket_logp)

# `Product`s can be treated as an array of iid variables
conditioned(d0::Product, blanket_logp) = Product(conditioned.(d0.v, blanket_logp))
function conditioned(d0::Product, blanket_logp, ix)
    # apply blanket accumulation to a subset of the product distribution

    ix_cartesian = CartesianIndex(ix)
    p = Product([(ix_cartesian == i) ? conditioned(d, blanket_logp) : d
                 for (i, d) in pairs(IndexCartesian(), d0.v)])
    return p
end

conditioned(d0::Distribution, blanket_logps) =
    throw(ArgumentError("Cannot condition a non-discrete or non-univariate distribution $d0."))


# from https://github.com/JuliaStats/StatsFuns.jl/blob/master/src/basicfuns.jl#L259
function softmax!(x::AbstractArray{<:AbstractFloat})
    u = maximum(x)
    s = zero(u)
    
    @inbounds for i in eachindex(x)
        s += (x[i] = exp(x[i] - u))
    end
    
    s⁻¹ = inv(s)
    
    @inbounds for i in eachindex(x)
        x[i] *= s⁻¹
    end
    
    return x
end

softmax(x::AbstractArray{<:AbstractFloat}) = softmax!(copy(x))



# D_w = Dirichlet(2, 1.0)
# w = rand(D_w)
# D_p = DiscreteNonParametric([0.3, 0.6], w)
# p = rand(D_p)
# D_x = Bernoulli(p)
# x = rand(D_x)
# conditioned(D_p, [logpdf(D_x, x)])
