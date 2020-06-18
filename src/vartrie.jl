using DataStructures: MutableLinkedList
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
        append!(trie.values, value)
        data = @view trie.values[(end - length(value) + 1):end]
        trie.nodes[sym] = ValueNode(vn, reshape(data, size(value)))
    end

    return trie
end

function Base.setindex!(trie::VarTrie, value, vn::VarName{sym}) where {sym}
    if haskey(trie.nodes, sym)
        setindex!(trie.nodes[sym], value, DynamicPPL.getindexing(vn)...)
    else
        # node = SubArrayNode(PartitionNode())
        error("not implemented")
    end

    return trie
end

Base.getindex(trie::VarTrie, vn::VarName{sym}) where {sym} =
    getindex(trie.nodes[sym], DynamicPPL.getindexing(vn)...)


Base.keys(trie::VarTrie) = mapfoldl(keys, append!, values(trie.nodes); init=Vector{VarName}())

# this can probably not be just replaced by trie.values, since the same order as `keys` should
# be ensured, and we want a copy anyway.
Base.values(trie::VarTrie{T}) where {T} = mapfoldl(values, push!, values(trie.nodes);
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


struct ValueNode{T, N, TVn<:VarName, TArr<:AbstractArray{T, N}} <: IndexNode{T}
    vn::TVn
    vals::TArr
end

Base.keys(node::ValueNode) = VarName[node.vn]
Base.values(node::ValueNode{T}) where {T} = T[node.vals;]
Base.values(node::ValueNode{T, 0}) where {T} = first(node.vals)
Base.getindex(node::ValueNode) = values(node)
Base.getindex(node::ValueNode, index) = getindex(node.vals, index...)
Base.setindex!(node::ValueNode, value) = (copyto!(node.vals, value); node)
Base.setindex!(node::ValueNode, value, index) = (setindex!(node.vals, value, index...); node)
Base.show(io::IO, node::ValueNode) = print(io, values(node))


struct PartitionNode{T, TList<:MutableLinkedList{<:IndexNode{T}}}
    indices::TList
    children::TList
end

# function Base.getindex(node::PartitionNode, index, indexing...)
    # for (i, ix) in enumerate(node.indices)
        # if issubset(ix, index)
            
        # end
    # end

    # throw(BoundsError(node, index))
# end

# isleft(::Tuple{}, ::Tuple{}) = true
# isleft(t1::NTuple{N}, t2::NTuple{N}) where {N} = isleft


struct SubArrayNode
    partition::PartitionNode
end
