abstract type IndexNode{T} end


struct VarTrie{T, TVal<:AbstractVector{T}}
    nodes::Dict{Symbol, IndexNode{T}}
    values::TVal
end

VarTrie{T}() where {T} = VarTrie(Dict{Symbol, IndexNode{T}}(), Vector{T}())


_defaultnode(value::AbstractArray) = ArrayNode(value)
_defaultnode(value) = ScalarNode(value)

function Base.setindex!(trie::VarTrie, value, vn::VarName{sym, Tuple{}}) where {sym}
    if haskey(trie.nodes, sym)
        setindex!(trie.nodes[sym], value)
    else
        trie.nodes[sym] = _defaultnode(value)
    end
end

function Base.setindex!(trie::VarTrie, value, vn::VarName{sym}) where {sym}
    indexing = DynamicPPL.getindexing(vn)
    if haskey(trie.nodes, sym)
        setindex!(trie.nodes[sym], value)
    else
        branch = BranchNode(first(indexing))
        setindexed!(branch, value, Base.tail(indexing))
        trie.nodes[sym] = branch
    end
end


Base.getindex(trie::VarTrie, vn::VarName{sym, Tuple{}}) = getindexed(trie.nodes[sym])
Base.getindex(trie::VarTrie, vn::VarName{sym}) where {sym} =
    getindexed(trie.nodes[sym], DynamicPPL.getindexing(vn))


Base.eachindex(trie::VarTrie) = mapreduce(eachindex, append!, trie.nodes; init=Vector{VarName}())
Base.IteratorSize(::Type{<:VarTrie}) = Base.HasLength()
Base.length(trie::VarTrie) = length(trie.nodes)
Base.IteratorEltype(::Type{<:VarTrie{T}}) = HasEltype()
Base.eltype(trie::VarTrie{T}) where {T} = Pair{VarName, T}
# Base.iterate(trie::VarTrie) = iterate(vn => trie[vn] for vn in keys(trie.nodes))
# Base.iterate(trie::VarTrie, state)


# function Base.show(io::IO, trie::VarTrie{T}) where {T}
#     L = length(trie.nodes)
#     if L == 0
#         print(io, "VarTrie{$T} with 0 entries")
#     else
#         println(io, "VarTrie{$T} with $(L) $(L == 1 ? "entry" : "entries"):")
#         for (k, v) in trie.nodes
#             if v isa ScalarNode
#                 println(io, "\t $k => $(v.value[])")
#             elseif v isa ArrayNode
#                 range = UnitRange(eachindex(v.value))
#                 println(io, "\t $k[$range] => $(v.value)")
#             elseif v isa BranchNode
#                 println(io, "\t $k => <<branch node>>")
#             end
#         end
#     end
# end


struct ScalarNode{T, TVn<:VarName, TVal<:AbstractVector{T}} <: IndexNode{T}
    vn::TVn
    values::TVal
end

getindexed(node::ScalarNode) = first(node.values)
Base.getindex(node::ScalarNode) = first(node.values)
Base.setindex!(node::ScalarNode, value) = node.values .= value
Base.eachindex(node::ScalarNode) = VarName[node.vn]


struct ArrayNode{T, N, TVn<:VarName, TArr<:AbstractArray{T, N}} <: IndexNode{T}
    vn::TVn
    value::TArr
end

getindexed(node::ArrayNode) = array.value
Base.getindex(node::ArrayNode, index...) = node.values[index]
Base.setindex!(node::ArrayNode, value, index...) = node.values[index] .= value
Base.eachindex(node::ArrayNode) = VarName[node.vn]





struct BranchNode{T, N, TChildren<:IndexNode{T}} <: IndexNode{T}
    children::Array{TChildren, N}
end



function Base.setindex!(node::ScalarNode, value)
    node.value[] = value
end

function Base.setindex!(node::ArrayNode, value, indices...)
    node.value[indices] = value
end

function Base.setindex!(node::BranchNode, value, indices...)
    setindex!(node.children, value, indices...)
end


