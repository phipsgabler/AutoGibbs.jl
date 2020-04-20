using DynamicPPL
using Distributions

export strip_dependencies, makegraph, showgraph, trackdependencies


const CallingNode{F} = Union{PrimitiveCallNode{<:Any, F}, NestedCallNode{<:Any, F}}


istilde(::CallingNode{typeof(DynamicPPL.tilde_assume)}) = true
istilde(::CallingNode{typeof(DynamicPPL.tilde_observe)}) = true
istilde(::CallingNode{typeof(DynamicPPL.dot_tilde_assume)}) = true
istilde(::CallingNode{typeof(DynamicPPL.dot_tilde_observe)}) = true
istilde(::AbstractNode) = false


function tilde_parameters(node::CallingNode{typeof(DynamicPPL.tilde_assume)})
    args = node.call.arguments # ctx, sampler, right, vn, inds, vi
    vn, dist, value = args[4], args[3], getvalue(node)
    return vn, dist, TapeConstant(value)
end

function tilde_parameters(node::CallingNode{typeof(DynamicPPL.tilde_observe)})
    args = node.call.arguments
    if length(args) == 7
        # ctx, sampler, right, left, vname, vinds, vi
        vn, dist, value = args[5], args[3], args[4]
        return vn, dist, value
    elseif length(args) == 5
        # ctx, sampler, right, left, vi
        dist, value = args[3], args[4]
        return nothing, dist, value
    end
end

function tilde_parameters(node::CallingNode{typeof(DynamicPPL.dot_tilde_assume)})
    args = node.call.arguments # ctx, sampler, right, left, vn, inds, vi
    vn, dist, value = args[5], args[3], args[4]
    return vn, dist, value
end

function tilde_parameters(node::CallingNode{typeof(DynamicPPL.dot_tilde_observe)})
    args = node.call.arguments
    if length(args) == 7
        # ctx, sampler, right, left, vn, inds, vi
        vn, dist, value = args[5], args[3], args[4]
        return vn, dist, value
    elseif length(args) == 5
        # ctx, sampler, right, left, vi
        dist, value = args[3], args[4]
        return nothing, dist, value
    end
end

tilde_parameters(node::AbstractNode) = nothing


# function dist_backward!(deps, node::AbstractNode)
#     current_refs = Vector{AbstractNode}(referenced(node))
#     push!(deps, node)

#     while !isempty(current_refs)
#         node = pop!(current_refs)
#         istilde(node) && continue
        
#         new_refs = referenced(node)
#         push!(deps, node)
#         union!(current_refs, new_refs)
#     end

#     return deps
# end


# function dist_forward!(deps, node::AbstractNode)
#     current_refs = Vector{AbstractNode}(dependents(node))
#     push!(deps, node)

#     while !isempty(current_refs)
#         node = pop!(current_refs)
#         istilde(node) && continue
        
#         new_refs = dependents(node)
#         push!(deps, node)
#         union!(current_refs, new_refs)
#     end

#     return deps
# end

# function strip_dependencies(node::NestedCallNode)
#     deps = Vector{AbstractNode}()
#     for child in getchildren(node)
#         bb = tilde_parameters(child)
#         if !isnothing(bb)
#             vn, dist, value = bb
#             push!(deps, child)
#             if dist isa TapeReference
#                 dist_backward!(deps, dist[])
#             end
#             if value isa TapeReference
#                 dist_backward!(deps, value[])
#             end
#         end
#     end

#     return sort(deps; by=n -> getposition(n.info)) |> unique
# end


function model_argument_deps(node)
    # From the beginning of the trace,
    # ```
    # @1: [Arg:§1:%1] @9#1 = ##evaluator#462
    # @2: [Arg:§1:%2] @9#2 = Model{...}
    # @3: [Arg:§1:%3] @9#3 = VarInfo (1 variable (s), dimension 1; logp: -3.957)
    # @4: [Arg:§1:%4] @9#4 = SampleFromPrior()
    # @5: [Arg:§1:%5] @9#5 = DefaultContext()
    # @6: [§1:%6] ⟨getproperty⟩(@2, ⟨:args⟩) = (x = [0.5, 1.1],)
    # @7: [§1:%7] ⟨getproperty⟩(@6, ⟨:x⟩) = [0.5, 1.1]
    # @8: [§1:%8] ⟨DynamicPPL.matchingvalue⟩(@4, @3, @7) = [0.5, 1.1]
    # ```
    # extract only the `getproperty(@6, ⟨:x⟩)` line (for each argument).

    deps = Vector{AbstractNode}()
    model_node = getchildren(node)[2]
    args_nodes = Vector{AbstractNode}()
    
    for child in getchildren(node)
        if child isa CallingNode{typeof(getproperty)}
            if getvalue(child.call.arguments[2]) == :args && child.call.arguments[1][] == model_node
                push!(args_nodes, child)
            elseif child.call.arguments[1][] ∈ args_nodes
                push!(deps, child)
            end
        end
    end

    return deps
end

function add_candidates!(dependencies, candidates, mutants, node)
    # all all candidates that are backwards references of this node to the proper dependencies
    current_refs = Vector{AbstractNode}(referenced(node))
    push!(dependencies, node)

    while !isempty(current_refs)
        node = pop!(current_refs)
        @show node, 1
        node ∈ dependencies && continue
        @show node, 1
        
        if node ∈ candidates
            push!(dependencies, node)
            union!(current_refs, referenced(node))
        end
        
        if haskey(mutants, node)
            @show mutants[node]
            push!(dependencies, node)
            union!(current_refs, mutants[node])
        end
    end
end



const MutatingFunctions = Union{typeof(setindex!), typeof(push!)}

function strip_dependencies(node::NestedCallNode)
    # Slice out only those nodes that are between tildes or model arguments,
    # taking some care to get mutated things (vector + `setindex!` etc.) right.

    # "dependencies" are nodes that are in the dependency graph for sure.
    # "candidates" are nodes that follow a dependency node; are included as dependencies, if the
    # chain ends up in a dependency node again (e.g., x ~ D, y = f(x), z ~ D2(y)).
    # "mutants" are backedges (from mutated to mutators) in the dependency graph resulting from
    # later operations mutating earlier values(e.g, x = zeros(), x[1] ~ D)
    
    dependencies = model_argument_deps(node)
    # dependencies = Vector{AbstractNode}()
    candidates = Set{AbstractNode}()
    mutants = Dict{AbstractNode, Vector{AbstractNode}}()

    for child in getchildren(node)
        info = tilde_parameters(child)
        if !isnothing(info)
            # tilde node: is a dependency for sure
            push!(dependencies, child)

            # Go backward and make all candidates between this and other dependencies
            # dependencies as well
            vn, dist, value = info
            dist isa TapeReference && add_candidates!(dependencies, candidates, mutants, dist[])
            value isa TapeReference && add_candidates!(dependencies, candidates, mutants, value[])
        else
            # non-tilde node: make candidate, if any parent is a candidate
            if any(r ∈ candidates || r ∈ dependencies for r in referenced(child))
                push!(candidates, child)
                
                if child isa CallingNode{<:MutatingFunctions}
                    # if the candidate is mutating, also make the mutated object a candidate
                    # and record the backedge from the mutated one to 
                    mutated = child.call.arguments[1]
                    if mutated isa TapeReference
                        push!(candidates, mutated[])
                        push!(get!(Vector{AbstractNode}, mutants, mutated[]), child)
                        @show mutants
                    end
                end
            end
        end
    end

    return sort(dependencies; by=n -> getposition(n.info)) |> unique
end













struct Reference
    name::Int
end

abstract type DepNode end

struct Assumption{T, TDist<:DepNode} <: DepNode
    vn::VarName
    dist::TDist
    value::T
end

struct Observation{T, TDist<:DepNode} <: DepNode
    vn::VarName
    dist::TDist
    value::T
end

struct Call{T, TF, TArgs<:Tuple} <: DepNode
    value::T
    f::TF
    args::TArgs
end

struct Constant{T} <: DepNode
    value::T
end

struct Argument{T} <: DepNode
    name::String
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
    vn_r, dist_r, value_r = tilde_parameters(node)
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

    graph[name] = tildetype(vn, dist, value)
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

# function pushnode!(graph, names, node::CallingNode{typeof(getindex)})
#     # special handling to get rid of the tuple result of `tilde_assume`
    
#     # array_ref = convertvalue(names, node.call.arguments[1])
#     # if graph[array_ref] isa Assumption
#     if node.call.arguments[1] isa CallingNode{<:Union{typeof(DynamicPPL.tilde_assume),
#                                                       typeof(DynamicPPL.dot_tilde_assume)}}
#         # when `getindex(x, ix...)` goes to an assumption `x`, then we overwrite the reference 
#         names[getposition(node)] = names[node.call.arguments[1].value]
#     else
#         # fall back to default case
#         invoke(pushnode!, Tuple{typeof(graph), typeof(names), CallingNode}, graph, names, node)
#     end
# end

function pushnode!(graph, names, node::CallingNode{typeof(DynamicPPL.matchingvalue)})
    # special handling for model arguments
    # @7: ⟨getproperty⟩(@6, ⟨:x⟩) = [0.5, 1.1]                                                                                                                                                       
    # @8: ⟨DynamicPPL.matchingvalue⟩(@4, @3, @7) = [0.5, 1.1]
    
    getproperty_ref = node.call.arguments[3]
    delete!(graph, names[getproperty_ref.index])
    
    argname = getvalue(getproperty_ref[].call.arguments[2])
    name = newname!(names, node)
    graph[name] = Argument(string(argname), getvalue(node))
end

function pushnode!(graph, names, node::ConstantNode)
    name = newname!(names, node)
    graph[name] = Constant(getvalue(node))
end

function pushnode!(graph, names, node::ArgumentNode)
    if !isnothing(node.branch_node)
        # skip branch argument nodes
        original = referenced(node)[1]
        names[getposition(node)] = names[getposition(original)]
    else
        # function argument nodes are handled like constants
        name = newname!(names, node)
        graph[name] = Constant(getvalue(node))
    end
end


convertvalue(names, value::TapeReference) = get(names, value.index, getvalue(value))
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
    dependency_slice = strip_dependencies(strip_model_layers(trace))
    return makegraph(dependency_slice)
end


Base.show(io::IO, r::Reference) = print(io, "%", r.name)
Base.show(io::IO, expr::Union{Assumption, Observation}) =
    print(io, expr.vn, " ~ ", expr.dist.f, "(", join(expr.dist.args, ", "), ") = ", expr.value)
Base.show(io::IO, expr::Call) = print(io, expr.f, "(", join(expr.args, ", "), ") = ", expr.value)
Base.show(io::IO, expr::Constant) = print(io, expr.value)
Base.show(io::IO, expr::Argument) = print(io, expr.name, " = ", expr.value)


function showgraph(graph)
    for (r, c) in sort(graph; by = ref -> ref.name)
        println(r, ": ", c)
    end
end





