using DynamicPPL
using Distributions

export strip_dependencies, makegraph, showgraph, trackdependencies


const CallingNode{F} = Union{PrimitiveCallNode{<:Any, F}, NestedCallNode{<:Any, F}}


istilde(::CallingNode{typeof(DynamicPPL.tilde_assume)}) = true
istilde(::CallingNode{typeof(DynamicPPL.tilde_observe)}) = true
istilde(::CallingNode{typeof(DynamicPPL.dot_tilde_assume)}) = true
istilde(::CallingNode{typeof(DynamicPPL.dot_tilde_observe)}) = true
istilde(::AbstractNode) = false


"""
    tilde_parameters(node)

Extract a tuple of variablen name, distribution, and value argument of any tilde call.  

The variable name can be `nothing` for constant observations.  The whole value is `nothing` for 
nodes that are not tilde calls.
"""
tilde_parameters(node::AbstractNode) = nothing

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


"""
    model_argument_nodes(root)

Extract from a stripped model trace all model argument `getindex` calls.
"""
function model_argument_nodes(root)
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

    argument_nodes = Vector{AbstractNode}()
    model_node = getchildren(root)[2]
    modelargs_nodes = Vector{AbstractNode}()
    
    for child in getchildren(root)
        if child isa CallingNode{typeof(getproperty)}
            if getvalue(child.call.arguments[2]) == :args && child.call.arguments[1][] == model_node
                push!(modelargs_nodes, child)
            elseif child.call.arguments[1][] ∈ modelargs_nodes
                # a `getproperty(args, symbol)` whose parent is a `args = getproperty(model, :args)`
                # is a model argument, for sure
                push!(argument_nodes, child)
            end
        end
    end

    return argument_nodes
end


"""
    add_candidates!(dependencies, candidates, mutants, node)

Traverse refernces of `node` backwards and turn the reached `candidates` into `dependencies`.
"""
function add_candidates!(dependencies, candidates, mutants, node)
    current_refs = Vector{AbstractNode}(referenced(node))
    push!(dependencies, node)

    while !isempty(current_refs)
        node = pop!(current_refs)
        node ∈ dependencies && continue
        
        if node ∈ candidates
            push!(dependencies, node)
            union!(current_refs, referenced(node))
        end
        
        if haskey(mutants, node)
            push!(dependencies, node)
            union!(current_refs, mutants[node])
        end
    end
end



"""
    strip_dependencies(root)

Slice out only those nodes that are between tildes or model arguments, taking some care to get
mutated things (vector + `setindex!` etc.) right.
"""
function strip_dependencies(root)
    # "dependencies" are nodes that are in the dependency graph for sure (i.e., tildes).
    # "candidates" are nodes that follow a dependency node; are included as dependencies, if the
    # chain ends up in a dependency node again (e.g., x ~ D, y = f(x), z ~ D2(y)) -- this is
    # what `add_candidates!` does by working backwards.
    # "mutants" are backedges (from mutated to mutators) in the dependency graph, resulting from
    # later operations mutating earlier values(e.g, z = zeros(), z[1] ~ D, x ~ D2(z[1])).
    
    dependencies = model_argument_nodes(root)
    # dependencies = Vector{AbstractNode}()
    candidates = Set{AbstractNode}()
    mutants = Dict{AbstractNode, Vector{AbstractNode}}()

    for child in getchildren(root)
        info = tilde_parameters(child)
        if !isnothing(info)
            # tilde node: is a dependency for sure
            push!(dependencies, child)
        else
            # non-tilde node: make candidate, if any parent is a candidate or dependency
            if any(r ∈ candidates || r ∈ dependencies for r in referenced(child))
                push!(candidates, child)
                
                if child isa CallingNode{typeof(setindex!)}
                    # if the candidate is mutating an array, also make the mutated object a candidate
                    # and record the backedge from the mutated one to:
                    # `n = setindex!(x, v, i)` makes `x` a candidate, and a backedge from `x` to `n`.
                    mutated = child.call.arguments[1]
                    if mutated isa TapeReference
                        push!(candidates, mutated[])
                        push!(get!(Vector{AbstractNode}, mutants, mutated[]), child)
                    end
                end
            end
        end
    end

    # Go over nodes once more and turn candidates between dependencies into dependencies, too.
    for child in getchildren(root)
        info = tilde_parameters(child)
        if !isnothing(info)
            vn, dist, value = info
            dist isa TapeReference && add_candidates!(dependencies, candidates, mutants, dist[])
            value isa TapeReference && add_candidates!(dependencies, candidates, mutants, value[])
        end
    end

    return sort(dependencies; by=getposition) |> unique
end



abstract type DepNode end

struct Assumption{T, TDist<:DepNode} <: DepNode
    dist::TDist
    value::T
end

struct Observation{T, TDist<:DepNode} <: DepNode
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


shortname(d::Type{<:Distribution}) = string(nameof(d))
shortname(other) = string(other)

Base.show(io::IO, expr::Assumption) =
    print(io, shortname(expr.dist.f), "(", join(expr.dist.args, ", "), ") → ", expr.value)
Base.show(io::IO, expr::Observation) =
    print(io, shortname(expr.dist.f), "(", join(expr.dist.args, ", "), ") → ", expr.value)
Base.show(io::IO, expr::Call) = print(io, expr.f, "(", join(expr.args, ", "), ") → ", expr.value)
Base.show(io::IO, expr::Constant) = print(io, expr.value)
Base.show(io::IO, expr::Argument) = print(io, expr.name, " → ", expr.value)


struct Reference
    number::Int
end

Base.show(io::IO, r::Reference) = print(io, "%", r.number)
Base.isless(q::Reference, r::Reference) = isless(q.number, r.number)
Base.hash(r::Reference, h::UInt) = hash(r.number, h)
Base.:(==)(q::Reference, r::Reference) = q.number == r.number


struct Graph
    statements::Dict{Union{Reference, VarName}, DepNode}
    reference_mapping::Dict{AbstractNode, Reference}
end

Graph() = Graph(Dict{Reference, DepNode}(), Dict{AbstractNode, Reference}())

Base.IteratorSize(::Type{Graph}) = Base.HasLength()
Base.length(graph::Graph) = length(graph.statements)
Base.IteratorEltype(::Type{Graph}) = Base.HasEltype()
Base.eltype(graph::Graph) = eltype(graph.statements)
Base.getindex(graph::Graph, ref) = graph.statements[ref]
Base.setindex!(graph::Graph, depnode, ref) = graph.statements[ref] = depnode
Base.haskey(graph::Graph, ref) = haskey(graph.statements, ref)
Base.delete!(graph::Graph, ref) = delete!(graph.statements, ref)

function _makewrappediter(graph::Graph)
    stmts = sort(graph.statements)
    return stmts, iterate(stmts)
end

function Base.iterate(graph::Graph, (stmts, iter)=_makewrappediter(graph))
    if !isnothing(iter)
        el, state = iter
        return el, (stmts, iterate(stmts, state))
    else
        return nothing
    end
end


getmapping(graph::Graph, node) = graph.reference_mapping[node]
setmapping!(graph::Graph, (node, ref)::Pair) = graph.reference_mapping[node] = ref

function makereference!(graph, node)
    newref = Reference(length(graph.reference_mapping) + 1)
    setmapping!(graph, node => newref)
    return newref
end

function Base.show(io::IO, graph::Graph)
    for (ref, dnode) in graph
        if ref isa Reference
            println(io, ref, " = ", dnode)
        else
            println(io, ref, " ~ ", dnode)
        end
    end
end



convertvalue(graph, texpr::TapeReference) = get(graph.reference_mapping, texpr[], getvalue(texpr))
convertvalue(graph, texpr::TapeConstant) = getvalue(texpr)


function makecallnode(graph, node::CallingNode)
    f = convertvalue(graph, node.call.f)
    args = convertvalue.(Ref(graph), (node.call.arguments..., something(node.call.varargs, ())...))
    return Call(getvalue(node), f, args)
end


function pushtilde!(graph, callingnode, maketilde)
    vn_r, dist_r, value_r = tilde_parameters(callingnode)
    vn = getvalue(vn_r)
    
    if dist_r isa TapeReference
        ref = getmapping(graph, dist_r[])
        if haskey(graph, ref)
            # move the separeate distribution call into the node and delete it from the graph
            dist = graph[ref]
            delete!(graph, ref)
        else
            # the distribution node existed, but has already been deleted, so we reconstruct it
            # (obscure case when one distribution reference is sampled from twice)
            dist = makecallnode(graph, dist_r[])
        end
    else
        # distribution was a constant in the IR
        dist = Constant(getvalue(dist_r))
    end

    value = convertvalue(graph, value_r)

    ref = makereference!(graph, callingnode)
    graph[ref] = maketilde(dist, value)
    return graph
end


function pushnode!(graph, node::CallingNode)
    ref = makereference!(graph, node)
    graph[ref] = makecallnode(graph, node)
    return graph
end

function pushnode!(graph, node::CallingNode{<:Union{typeof(DynamicPPL.tilde_assume),
                                                    typeof(DynamicPPL.dot_tilde_assume)}})
    return pushtilde!(graph, node, Assumption)
end

function pushnode!(graph, node::CallingNode{<:Union{typeof(DynamicPPL.tilde_observe),
                                                    typeof(DynamicPPL.dot_tilde_observe)}})
    return pushtilde!(graph, node, Observation)
end

function pushnode!(graph, node::CallingNode{typeof(DynamicPPL.matchingvalue)})
    # special handling for model arguments
    # @7: ⟨getproperty⟩(@6, ⟨:x⟩) = [0.5, 1.1]                                                                                                                                                       
    # @8: ⟨DynamicPPL.matchingvalue⟩(@4, @3, @7) = [0.5, 1.1]
    
    getproperty_node = node.call.arguments[3][]
    # delete!(graph, getmapping(graph, getproperty_node))
    
    argname = getvalue(getproperty_node.call.arguments[2])
    ref = makereference!(graph, node)
    graph[ref] = Argument(string(argname), getvalue(node))
    return graph
end

function pushnode!(graph, node::ConstantNode)
    ref = makereference!(graph, node)
    graph[ref] = Constant(getvalue(node))
    return graph
end

function pushnode!(graph, node::ArgumentNode)
    if !isnothing(node.branch_node)
        # skip branch argument nodes
        original_ref = getmapping(graph, referenced(node)[1])
        setmapping!(graph, node => original_ref)
    else
        # function argument nodes are handled like constants
        ref = makereference!(graph, node)
        graph[ref] = Constant(getvalue(node))
    end

    return graph
end


function makegraph(slice::Vector{<:AbstractNode})
    return foldl(pushnode!, slice, init=Graph())
end


function trackdependencies(model)
    trace = trackmodel(model)
    dependency_slice = strip_dependencies(strip_model_layers(trace))
    return makegraph(dependency_slice)
end





