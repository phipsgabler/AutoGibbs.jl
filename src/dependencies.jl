using DynamicPPL

export dependencies


const CallingNode{F} = Union{PrimitiveCallNode{<:Any, F}, NestedCallNode{<:Any, F}}


function extract(node::CallingNode{typeof(DynamicPPL.tilde_assume)})
    args = node.call.arguments # ctx, sampler, right, vn, inds, vi
    vn, dist, (value, logp) = args[4], args[3], getvalue(node)
    return vn, dist, value, logp
end

function extract(node::CallingNode{typeof(DynamicPPL.tilde_observe)})
    args = node.call.arguments
    if length(args) == 7
        # ctx, sampler, right, left, vname, vinds, vi
        vn, dist, value = args[5], args[3], getvalue(args[4])
        return vn, dist, value, nothing
    elseif length(args) == 5
        # ctx, sampler, right, left, vi
        dist, value = args[3], getvalue(args[4])
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
        vn, dist, value = args[5], args[3], getvalue(args[4])
        return vn, dist, value, nothing
    elseif length(args) == 5
        # ctx, sampler, right, left, vi
        dist, value = args[3], getvalue(args[4])
        return nothing, dist, value, nothing
    end
end

extract(node::AbstractNode) = nothing

function dependencies(node::NestedCallNode)
    deps = []
    for child in getchildren(node)
        bb = extract(child)
        if !isnothing(bb)
            vn, dist, value, logp = bb
            if !isnothing(vn)
                push!(deps, getvalue(vn) => (getvalue(dist), value, logp))
            end
        end
    end

    return deps
end
