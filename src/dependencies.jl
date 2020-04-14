using DynamicPPL
using Distributions

export dependencies


const CallingNode{F} = Union{PrimitiveCallNode{<:Any, F}, NestedCallNode{<:Any, F}}


istilde(::CallingNode{typeof(DynamicPPL.tilde_assume)}) = true
istilde(::CallingNode{typeof(DynamicPPL.tilde_observe)}) = true
istilde(::CallingNode{typeof(DynamicPPL.dot_tilde_assume)}) = true
istilde(::CallingNode{typeof(DynamicPPL.dot_tilde_observe)}) = true
istilde(::AbstractNode) = false


function extract(node::CallingNode{typeof(DynamicPPL.tilde_assume)})
    args = node.call.arguments # ctx, sampler, right, vn, inds, vi
    vn, dist, (value, logp) = args[4], args[3], getvalue(node)
    return vn, dist, value, logp
end

function extract(node::CallingNode{typeof(DynamicPPL.tilde_observe)})
    args = node.call.arguments
    if length(args) == 7
        # ctx, sampler, right, left, vname, vinds, vi
        vn, dist, value = args[5], args[3], args[4]
        return vn, dist, value, nothing
    elseif length(args) == 5
        # ctx, sampler, right, left, vi
        dist, value = args[3], args[4]
        return nothing, dist, value, nothing
    end
end

function extract(node::CallingNode{typeof(DynamicPPL.dot_tilde_assume)})
    args = node.call.arguments # ctx, sampler, right, left, vn, inds, vi
    vn, dist, (value, logp) = args[5], args[3], getvalue(node)
    return vn, dist, value, logp
end

function extract(node::CallingNode{typeof(DynamicPPL.dot_tilde_observe)})
    args = node.call.arguments
    if length(args) == 7
        # ctx, sampler, right, left, vn, inds, vi
        vn, dist, value = args[5], args[3], args[4]
        return vn, dist, value, nothing
    elseif length(args) == 5
        # ctx, sampler, right, left, vi
        dist, value = args[3], args[4]
        return nothing, dist, value, nothing
    end
end

extract(node::AbstractNode) = nothing


function dist_backward!(deps, node::AbstractNode)
    current_refs = Vector{AbstractNode}(referenced(node))
    push!(deps, node)
    
    while !isempty(current_refs)
        node = pop!(current_refs)
        istilde(node) && continue
        
        new_refs = referenced(node)
        push!(deps, node)
        union!(current_refs, new_refs)
    end

    return deps
end

function dependencies(node::NestedCallNode)
    deps = Vector{AbstractNode}()
    for child in getchildren(node)
        bb = extract(child)
        if !isnothing(bb)
            vn, dist, value, logp = bb
            push!(deps, child)
            if dist isa TapeReference
                dist_backward!(deps, dist[])
            end
        end
    end

    return sort(deps; by=n -> getposition(n.info)) |> unique
end
