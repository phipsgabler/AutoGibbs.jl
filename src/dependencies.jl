using DynamicPPL
using Distributions

export dependencies, makegraph, showgraph, trackdependencies


const CallingNode{F} = Union{PrimitiveCallNode{<:Any, F}, NestedCallNode{<:Any, F}}


istilde(::CallingNode{typeof(DynamicPPL.tilde_assume)}) = true
istilde(::CallingNode{typeof(DynamicPPL.tilde_observe)}) = true
istilde(::CallingNode{typeof(DynamicPPL.dot_tilde_assume)}) = true
istilde(::CallingNode{typeof(DynamicPPL.dot_tilde_observe)}) = true
istilde(::AbstractNode) = false


function extract(node::CallingNode{typeof(DynamicPPL.tilde_assume)})
    args = node.call.arguments # ctx, sampler, right, vn, inds, vi
    vn, dist, (value, logp) = args[4], args[3], getvalue(node)
    return vn, dist, TapeConstant(value), logp
end

function extract(node::CallingNode{typeof(DynamicPPL.tilde_observe)})
    args = node.call.arguments
    if length(args) == 7
        # ctx, sampler, right, left, vname, vinds, vi
        vn, dist, value, logp = args[5], args[3], args[4], getvalue(node)
        return vn, dist, value, logp
    elseif length(args) == 5
        # ctx, sampler, right, left, vi
        dist, value, logp = args[3], args[4], getvalue(node)
        return nothing, dist, value, logp
    end
end

function extract(node::CallingNode{typeof(DynamicPPL.dot_tilde_assume)})
    args = node.call.arguments # ctx, sampler, right, left, vn, inds, vi
    vn, dist, value, logp = args[5], args[3], args[4], getvalue(node)[2]
    return vn, dist, value, logp
end

function extract(node::CallingNode{typeof(DynamicPPL.dot_tilde_observe)})
    args = node.call.arguments
    if length(args) == 7
        # ctx, sampler, right, left, vn, inds, vi
        vn, dist, value, logp = args[5], args[3], args[4], getvalue(node)
        return vn, dist, value, logp
    elseif length(args) == 5
        # ctx, sampler, right, left, vi
        dist, value, logp = args[3], args[4], getvalue(node)
        return nothing, dist, value, logp
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






struct Reference
    name::Int
end

abstract type DepNode end

struct Assumption{T, TDist<:DepNode} <: DepNode
    vn::VarName
    dist::TDist
    value::T
    logp::Float64
end

struct Observation{T, TDist<:DepNode} <: DepNode
    vn::VarName
    dist::TDist
    value::T
    logp::Float64
end

struct Call{T, TF, TArgs<:Tuple} <: DepNode
    value::T
    f::TF
    args::TArgs
end

struct Constant{T} <: DepNode
    value::T
end


function newname!(names, node)
    newname = Reference(length(names) + 1)
    names[getposition(node)] = newname
    return newname
end
    

function makecallnode(names, node::CallingNode)
    f = convertvalue(names, node.call.f)
    args = convertvalue.(Ref(names), (node.call.arguments..., something(node.call.varargs, ())...))
    return Call(getvalue(node), f, args)
end


function pushtilde!(graph, names, node::CallingNode, tildetype)
    name = newname!(names, node)
    vn_r, dist_r, value_r, logp = extract(node)
    vn = getvalue(vn_r)
    if dist_r isa TapeReference
        ref = names[dist_r.index]
        if haskey(graph, ref)
            # move the separeate distribution call into the node and delete it from the graph
            dist = graph[ref]
            delete!(graph, ref)
        else
            # the distribution node existed, but has already been deleted, so we reconstruct it
            # (obscure case when one distribution reference is sampled from twice)
            dist = makecallnode(names, dist_r[])
        end
    else
        # distribution was a constant in the IR
        dist = Constant(getvalue(dist_r))
    end
    value = getvalue(value_r)

    graph[name] = tildetype(vn, dist, value, logp)
    return graph
end


function pushnode!(graph, names, node::CallingNode)
    name = newname!(names, node)
    graph[name] = makecallnode(names, node)
    return graph
end

function pushnode!(graph, names, node::CallingNode{<:Union{typeof(DynamicPPL.tilde_assume),
                                                           typeof(DynamicPPL.dot_tilde_assume)}})
    return pushtilde!(graph, names, node, Assumption)
end

function pushnode!(graph, names, node::CallingNode{<:Union{typeof(DynamicPPL.tilde_observe),
                                                           typeof(DynamicPPL.dot_tilde_observe)}})
    return pushtilde!(graph, names, node, Observation)
end

function pushnode!(graph, names, node::CallingNode{typeof(getindex)})
    # special handling to get rid of the tuple result of `tilde_assume`
    
    # array_ref = convertvalue(names, node.call.arguments[1])
    # if graph[array_ref] isa Assumption
    if node.call.arguments[1] isa CallingNode{<:Union{typeof(DynamicPPL.tilde_assume),
                                                      typeof(DynamicPPL.dot_tilde_assume)}}
        # when `getindex(x, ix...)` goes to an assumption `x`, then we overwrite the reference 
        names[getposition(node)] = names[node.call.arguments[1].value]
    else
        # fall back to default case
        invoke(pushnode!, Tuple{typeof(graph), typeof(names), CallingNode}, graph, names, node)
    end
end

function pushnode!(graph, names, node::ConstantNode)
    name = newname!(names, node)
    graph[name] = Constant(getvalue(node))
end

function pushnode!(graph, names, node::ArgumentNode)
    if !isnothing(node.branch_node)
        # skip branch argument nodes
        original = getparent(node)
        names[getposition(node)] = names[getposition(original)]
    else
        # function argument nodes are handled like constants
        name = newname!(names, node)
        graph[name] = Constant(getvalue(node))
    end
end


convertvalue(names, value::TapeReference) = names[value.index]
convertvalue(names, value::TapeConstant) = getvalue(value)

function makegraph(slice::Vector{<:AbstractNode})
    graph = Dict{Reference, DepNode}()  # resulting graph
    names = Dict{Int, Reference}()      # mapping from node indices to new references
    
    for node in slice
        pushnode!(graph, names, node)
    end

    return graph
end


function trackdependencies(model)
    trace = trackmodel(model)
    dependency_slice = dependencies(strip_model_layers(trace))
    return makegraph(dependency_slice)
end


Base.show(io::IO, r::Reference) = print(io, "%", r.name)
Base.show(io::IO, expr::Union{Assumption, Observation}) =
    print(io, expr.vn, " ~ ", expr.dist.f, "(", join(expr.dist.args, ", "), ") = ", expr.value)
Base.show(io::IO, expr::Call) = print(io, expr.f, "(", join(expr.args, ", "), ") = ", expr.value)
Base.show(io::IO, expr::Constant) = print(io, expr.value)


function showgraph(graph)
    for (r, c) in sort(graph; by = ref -> ref.name)
        println(r, ": ", c)
    end
end





