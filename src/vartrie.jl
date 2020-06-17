using DynamicPPL

export VarTrie


abstract type IndexNode{T} end


struct VarTrie{T, TVal<:AbstractVector{T}}
    nodes::Dict{Symbol, IndexNode{T}}
    values::TVal
end

VarTrie{T}() where {T} = VarTrie(Dict{Symbol, IndexNode{T}}(), Vector{T}())


function Base.setindex!(trie::VarTrie, value, vn::VarName{sym, Tuple{}}) where {sym}
    if haskey(trie.nodes, sym)
        setindex!(trie.nodes[sym], value)
    else
        push!(trie.values, value)
        trie.nodes[sym] = ScalarNode(vn, @view(trie.values[end:end]))
    end
end

function Base.setindex!(trie::VarTrie, value::AbstractArray, vn::VarName{sym, Tuple{}}) where {sym}
    if haskey(trie.nodes, sym)
        setindex!(trie.nodes[sym], value)
    else
        append!(trie.values, value)
        data = @view trie.values[(end - length(value) + 1):end]
        trie.nodes[sym] = ArrayNode(vn, reshape(data, size(value)))
    end
end

function Base.setindex!(trie::VarTrie, value, vn::VarName{sym}) where {sym}
    if haskey(trie.nodes, sym)
        setindex!(trie.nodes[sym], value, DynamicPPL.getindexing(vn)...)
    else
        error("create branch node")
    end
end

Base.getindex(trie::VarTrie, vn::VarName{sym}) where {sym} =
    getindex(trie.nodes[sym], DynamicPPL.getindexing(vn)...)


Base.keys(trie::VarTrie) = mapreduce(keys, append!, values(trie.nodes); init=Vector{VarName}())

# this can probably not be just replaced by trie.values, since the same order as `keys` should
# be ensured, and we want a copy anyway.
Base.values(trie::VarTrie{T}) where {T} = mapreduce(values, push!, values(trie.nodes);
                                                    init=Vector{Union{T, Array{T}}}())

Base.IteratorSize(::Type{<:VarTrie}) = Base.HasLength()
Base.length(trie::VarTrie) = length(trie.nodes)
Base.IteratorEltype(::Type{<:VarTrie{T}}) where {T} = HasEltype()
Base.eltype(trie::VarTrie{T}) where {T} = Pair{VarName, T}
# Base.iterate(trie::VarTrie) = iterate(vn => trie[vn] for vn in keys(trie.nodes))
# Base.iterate(trie::VarTrie, state)

# TODO: implement copy, copyto! for broadcasting


function Base.show(io::IO, trie::VarTrie{T}) where {T}
    L = length(trie.nodes)
    if L == 0
        print(io, "VarTrie{$T} with 0 entries")
    else
        println(io, "VarTrie{$T} with $(L) $(L == 1 ? "entry" : "entries"):")
        for (vn, val) in pairs(trie)
            println(io, " $vn => $val")
        end
    end
end


struct ScalarNode{T, TVn<:VarName, TVal<:AbstractVector{T}} <: IndexNode{T}
    vn::TVn
    vals::TVal
end

Base.keys(node::ScalarNode) = VarName[node.vn]
Base.values(node::ScalarNode) = first(node.vals)
Base.getindex(node::ScalarNode) = values(node)
Base.setindex!(node::ScalarNode, value) = setindex!(node.vals, value)
Base.show(io::IO, node::ScalarNode) = print(io, node[])


struct ArrayNode{T, N, TVn<:VarName, TArr<:AbstractArray{T, N}} <: IndexNode{T}
    vn::TVn
    vals::TArr
end

Base.keys(node::ArrayNode) = VarName[node.vn]
Base.values(node::ArrayNode{T}) where {T} = T[node.vals;]
Base.getindex(node::ArrayNode) = values(node)
Base.getindex(node::ArrayNode, index) = getindex(node.vals, index...)
Base.setindex!(node::ArrayNode, value) = setindex!(node.vals, value, :)
Base.setindex!(node::ArrayNode, value, index) = setindex!(node.vals, value, index...)
Base.show(io::IO, node::ArrayNode) = print(io, values(node))




# struct BranchNode{T, N, TChildren<:IndexNode{T}} <: IndexNode{T}
#     children::Array{TChildren, N}
# end

# function Base.setindex!(node::BranchNode, value, indices...)
#     setindex!(node.children, value, indices...)
# end
