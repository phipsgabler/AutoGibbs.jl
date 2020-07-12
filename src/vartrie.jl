using DynamicPPL


export VarTrie



abstract type IndexNode{T, N} end



struct VarTrie{T, TVal<:AbstractVector{T}}
    nodes::Dict{Symbol, IndexNode{T}}
    data::TVal
end

VarTrie{T}() where {T} = VarTrie(Dict{Symbol, IndexNode{T}}(), Vector{T}())


function _appendvalue!(trie, value, vn)
    append!(trie.data, value)
    data = @view trie.data[(end - length(value) + 1):end]
    return ValueNode(vn, reshape(data, size(value)))
end

function Base.setindex!(trie::VarTrie, value, vn::VarName{sym, Tuple{}}) where {sym}
    if !haskey(trie.nodes, sym)
        trie.nodes[sym] = _appendvalue!(trie, value, vn)
    else
        setindex!(trie.nodes[sym], value; trie=trie, vn=vn)
    end

    return trie
end

function Base.setindex!(trie::VarTrie, value, vn::VarName{sym}) where {sym}
    indexing = DynamicPPL.getindexing(vn)
    
    if !haskey(trie.nodes, sym)
        trie.nodes[sym] = SubArrayNode{valuetype(trie), length(first(indexing))}()
    end

    setindex!(trie.nodes[sym], value, indexing...; trie=trie, vn=vn)

    return trie
end




Base.getindex(trie::VarTrie, vn::VarName{sym}) where {sym} =
    getindex(trie.nodes[sym], DynamicPPL.getindexing(vn)...)


Base.keys(trie::VarTrie) = mapfoldl(keys, append!, values(trie.nodes); init=Vector{VarName}())

# this can probably not be just replaced by trie.data, since the same order as `keys` should
# be ensured, and we want a copy anyway.
Base.values(trie::VarTrie{T}) where {T} = mapfoldl(values, push!, values(trie.nodes);
                                                   init=Vector{Union{T, Array{T}}}())

Base.IteratorSize(::Type{<:VarTrie}) = Base.HasLength()
Base.length(trie::VarTrie) = length(trie.nodes)
Base.IteratorEltype(::Type{<:VarTrie{T}}) where {T} = HasEltype()
Base.eltype(trie::VarTrie{T}) where {T} = Pair{VarName, T}
# Base.iterate(trie::VarTrie) = iterate(vn => trie[vn] for vn in keys(trie.nodes))
# Base.iterate(trie::VarTrie, state)


valuetype(::Type{<:VarTrie{T}}) where {T} = T
valuetype(trie::VarTrie) = valuetype(typeof(trie))

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



struct ValueNode{T, N, TVn<:VarName, TArr<:AbstractArray{T, N}} <: IndexNode{T, N}
    vn::TVn
    vals::TArr
end

Base.keys(node::ValueNode) = VarName[node.vn]
Base.values(node::ValueNode{T}) where {T} = T[node.vals;]
Base.values(node::ValueNode{T, 0}) where {T} = first(node.vals)  # dimension 0 nodes are treated as scalars
Base.getindex(node::ValueNode) = values(node)
Base.getindex(node::ValueNode, index...) = getindex(node.vals, index...)
Base.setindex!(node::ValueNode, value; trie=nothing, vn=nothing) =
    (copyto!(node.vals, value); node)
Base.setindex!(node::ValueNode, value, index...; trie=nothing, vn=nothing) =
    (setindex!(node.vals, value, index...); node)



struct PartitionIndex{N, I<:NTuple{N, Union{Int, AbstractUnitRange}}}
    index::I
    hash::UInt
end

PartitionIndex(index) = PartitionIndex(index, foldl((h, x) -> hash(x, h), index, init=zero(UInt)))

Base.show(io::IO, ix::PartitionIndex) = print(io, "PartitionIndex($(ix.index))")

Base.hash(ix::PartitionIndex, h::UInt) = hash(ix.hash, h)
Base.hash(ix::PartitionIndex) = ix.hash

Base.first(p::PartitionIndex) = first(p.index)
Base.tail(p::PartitionIndex) = PartitionIndex(Base.tail(p.index))

Base.isless(i1::PartitionIndex, i2::PartitionIndex) = i1.hash < i2.hash

Base.to_indices(A, I::Tuple{PartitionIndex}) = to_indices(A, I[1].index)



struct PartitionIndices{N, TArr<:Array{<:PartitionIndex{N}}}
    indices::TArr
end

PartitionIndices(index::Tuple) =
    PartitionIndices([PartitionIndex(ix) for ix in Iterators.ProductIterator(_wrapatomics.(index))])

_wrapatomics(i::Int) = (i,)
_wrapatomics(i::AbstractUnitRange) = (i,)
_wrapatomics(i) = i




# A `SubArrayNode` containing `PartitionNode`s is basically a B-Tree with cartesian indices as keys,
# ordered by hash, modulo ranges.  See Cormen et al., p. 287 ss.  We use a minimum degree of 2, 
# resulting in a 2-3-4 tree here.  Contversly to the book's design, leave nodes can have other
# index nodes as children for nested arrays.
const MinDegree = 2


abstract type PartitionNode{T, N} end

struct PartitionLeaf{T, N} <: PartitionNode{T, N}
    indices::Vector{PartitionIndex{N}} # T key elements
    children::Vector{IndexNode{T, N}} #  T value children pointers
end

PartitionLeaf{T, N}() where {T, N} = PartitionLeaf(Vector{NTuple{N}}(),
                                                   Vector{IndexNode{T, N}}())


struct PartitionBranch{T, N} <: PartitionNode{T, N}
    indices::Vector{PartitionIndex{N}} # T key elements
    children::Vector{PartitionNode{T, N}} # T + 1 children pointers
end

PartitionBranch{T, N}() where {T, N} = PartitionBranch(Vector{PartitionIndex{N}}(),
                                                       Vector{PartitionNode{T, N}}())

isleaf(::PartitionLeaf) = true
isleaf(::PartitionBranch) = false
Base.similar(n::PartitionLeaf, indices=empty(n.indices), children=empty(n.children)) =
    PartitionLeaf(indices, children)
Base.similar(n::PartitionBranch, indices=empty(n.indices), children=empty(n.children)) =
    PartitionBranch(indices, children)

Base.keys(node::PartitionNode) = mapfoldl(keys, append!, node.children; init=Vector{VarName}())
Base.values(node::PartitionNode{T}) where {T} = mapfoldl(values, push!, node.children;
                                                         init=Vector{Union{T, Array{T}}}())

function findpartition(node::PartitionNode, index)
    # B-tree search
    i = 1
    while i ≤ length(node.indices) && isleft(node.indices[i], index)
        i += 1
    end
    
    if i ≤ length(node.indices) && DynamicPPL._issubindex(node.indices[i], index)
        return node.children[i]
    elseif isleaf(node)
        return nothing
    else
        return findpartition(node.children[i], index)
    end
end

function splitpartition!(parent::PartitionNode, i::Int)
    # B-tree split child
    oldchild = parent.children[i] # y

    # move the t-1 largest children to the new child, z
    newchild = similar(oldchild,
                       oldchild.indices[MinDegree:end],
                       oldchild.children[MinDegree+1:end])
    
    # shrink old child
    resize!(oldchild.indices, MinDegree - 1)
    resize!(oldchild.children, min(length(oldchild.children), MinDegree))

    # insert new child into parent
    insert!(parent.children, i + 1, newchild)
    insert!(parent.indices, i, oldchild.indices[MinDegree - 1])

    return parent
end

function insertpartition!(node::PartitionNode, index)
    # B-tree insert nonfull

    i = length(node.indices)
    while i ≥ 1 && isless(index, node.indices[i])
        i -= 1
    end
    
    if isleaf(node)
        insert!(node.indices, i + 1, index)
    else
        i += 1
        if length(node.children[i].indices) == 2MinDegree - 1
            splitpartition!(node, i)
            isleft(node.indices[i], index) && (i += 1)
        end

        insertpartition!(node.children[i], index)
    end

    return node
end



mutable struct SubArrayNode{T, N} <: IndexNode{T, N}
    root::PartitionNode{T, N}
end

SubArrayNode{T, N}() where {T, N} = SubArrayNode(PartitionLeaf{T, N}())

Base.keys(node::SubArrayNode) = keys(node.root)
Base.values(node::SubArrayNode) = values(node.root)

function Base.getindex(node::SubArrayNode, index, tail...)
    partition_indices = PartitionIndices(index)
    partition_nodes = findpartitions(node, partition_indices)
    if !isnothing(partition_node)
        return [getindex(node, tail...) for node in partition_nodes]
    else
        throw(BoundsError(node, index))
    end
end

function Base.setindex!(node::SubArrayNode, value, index, tail...; trie, vn)
    partition_indices = PartitionIndices(index)
    partition_nodes = findpartitions(node, partition_indices)
    if !isnothing(partitions)
        setpartitions!(partition_nodes, value, tail...; trie=trie, vn=vn)
    else
        new_partitions = addpartitions!(node, partition_indices)
        setpartitions!(new_partitions, value, tail...; trie=trie, vn=vn)
    end

    return node
end

# function Base.setindex!(node::SubArrayNode, value; trie, vn)
#     value_nodes = findpartitions(node, partition_indices)
#     if !isnothing(value_nodes)
#         setpartitions!(value_nodes, value; trie=trie, vn=vn)
#     else
#         new_value = _appendvalue!(trie, value, vn)
#         addpartitions!([node], partition_indices)
#         setindex!(new_value, value)
#     end

#     return node
# end

# findpartition(node, index) = node
# findpartition(node::SubArrayNode, index) = findpartition(node.root, index)

function addpartition!(node::SubArrayNode{T, N}, index) where {T, N}
    # B-tree insert
    oldroot = node.root # r
    if length(oldroot.indices) == 2MinDegree - 1
        newroot = PartitionBranch{T, N}() # s
        node.root = newroot
        push!(newroot.children, oldroot)
        splitpartition!(newroot, 1)
        insertpartition!(newroot, index)
    else
        insertpartition!(oldroot, index)
    end
end





