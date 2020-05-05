using IRTracker
using DynamicPPL
using Distributions
using DataStructures: SortedDict

export strip_dependencies, makegraph, showgraph, trackdependencies


try_getindex(expr::TapeReference) = expr[]
try_getindex(expr::TapeConstant) = getvalue(expr)



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
    args = getarguments(node) # ctx, sampler, right, vn, inds, vi
    vn, dist, value = args[4], args[3], getvalue(node)
    return vn, dist, TapeConstant(value)
end

function tilde_parameters(node::CallingNode{typeof(DynamicPPL.tilde_observe)})
    args = getarguments(node)
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
    args = getarguments(node) # ctx, sampler, right, left, vn, inds, vi
    vn, dist, value = args[5], args[3], args[4]
    return vn, dist, value
end

function tilde_parameters(node::CallingNode{typeof(DynamicPPL.dot_tilde_observe)})
    args = getarguments(node)
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
            value, property_name = getarguments(child)
            if getvalue(property_name) == :args && try_getindex(value) == model_node
                push!(modelargs_nodes, child)
            elseif try_getindex(value) ∈ modelargs_nodes
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
        params = tilde_parameters(child)
        if !isnothing(params)
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
                    mutated = getargument(child, 1)
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
        params = tilde_parameters(child)
        if !isnothing(params)
            vn, dist, value = params
            vn isa TapeReference && add_candidates!(dependencies, candidates, mutants, vn[])
            dist isa TapeReference && add_candidates!(dependencies, candidates, mutants, dist[])
            value isa TapeReference && add_candidates!(dependencies, candidates, mutants, value[])
        end
    end

    return sort(dependencies; by=getposition) |> unique
end



abstract type Statement end

struct Assumption{dotted, TDist<:Statement, TV} <: Statement
    dist::TDist
    value::TV
end

Assumption{dotted}(dist, value) where {dotted} = Assumption{dotted, typeof(dist), typeof(value)}(dist, value)

struct Observation{dotted, TDist<:Statement, TV} <: Statement
    dist::TDist
    value::TV
end

Observation{dotted}(dist, value) where {dotted} = Observation{dotted, typeof(dist), typeof(value)}(dist, value)

struct Call{TF, TArgs<:Tuple, TV} <: Statement
    f::TF
    args::TArgs
    value::TV
end

struct Constant{TV} <: Statement
    value::TV
end



shortname(d::Type{<:Distribution}) = string(nameof(d))
shortname(other) = string(other)

Base.show(io::IO, stmt::Assumption) =
    print(io, shortname(stmt.dist.f), "(", join(stmt.dist.args, ", "), ") → ", stmt.value)
Base.show(io::IO, stmt::Observation) =
    print(io, shortname(stmt.dist.f), "(", join(stmt.dist.args, ", "), ") ← ", stmt.value)
Base.show(io::IO, stmt::Call) = print(io, stmt.f, "(", join(stmt.args, ", "), ") → ", stmt.value)
Base.show(io::IO, stmt::Constant) = print(io, stmt.value)

IRTracker.getvalue(stmt::Statement) = stmt.value

isdotted(::Assumption{dotted}) where {dotted} = dotted
isdotted(::Observation{dotted}) where {dotted} = dotted


struct Reference{TV<:Union{VarName, Nothing}}
    number::Int
    vn::TV
end

Reference(number) = Reference(number, nothing)

const UnnamedReference = Reference{Nothing}
const NamedReference = Reference{<:VarName}

Base.show(io::IO, r::UnnamedReference) = print(io, "⟨", r.number, "⟩")
Base.show(io::IO, r::NamedReference) = print(io, "⟨", r.number, ":", r.vn, "⟩")
Base.isless(q::Reference, r::Reference) = isless(q.number, r.number)
Base.hash(r::Reference, h::UInt) = hash(r.number, h)
Base.:(==)(q::Reference, r::Reference) = (q.number == r.number) #&& (q.vn == r.vn)


dependencies(::Constant) = Reference[]
function dependencies(stmt::Union{Assumption, Observation})
    direct_dependencies = dependencies(stmt.dist)
    stmt.value isa Reference && push!(direct_dependencies, stmt.value)
    return direct_dependencies
end
function dependencies(stmt::Call)
    direct_dependencies = Reference[arg for arg in stmt.args if arg isa Reference]
    return mapreduce(dependencies,
                     append!,
                     direct_dependencies,
                     init=direct_dependencies)
end
dependencies(ref::NamedReference) =
    Reference[ix for index in DynamicPPL.getindexing(ref.vn) for ix in index if ix isa Reference]
dependencies(ref::UnnamedReference) = Reference[]


struct Graph
    """Entries in graph: mapping from references to statements."""
    statements::SortedDict{Reference, Statement}

    """Association which nodes in the trace are associated with which references in the graph"""
    reference_mapping::Dict{AbstractNode, Reference}

    """Remembers which references describe arrays that have been discovered to have been mutated"""
    mutated_rvs::Dict{Reference, VarName}
end

Graph() = Graph(SortedDict{Reference, Statement}(),
                Dict{AbstractNode, Reference}(),
                Dict{Reference, VarName}())

Base.IteratorSize(::Type{Graph}) = Base.HasLength()
Base.length(graph::Graph) = length(graph.statements)
Base.IteratorEltype(::Type{Graph}) = Base.HasEltype()
Base.eltype(graph::Graph) = eltype(graph.statements)
Base.getindex(graph::Graph, ref::Reference) = graph.statements[ref]
Base.getindex(graph::Graph, ref::Int) = graph[Reference(ref)]
Base.setindex!(graph::Graph, stmt, ref) = graph.statements[ref] = stmt
Base.haskey(graph::Graph, ref) = haskey(graph.statements, ref)
Base.delete!(graph::Graph, ref) = delete!(graph.statements, ref)
Base.keys(graph::Graph) = keys(graph.statements)
Base.values(graph::Graph) = values(graph.statements)
Base.iterate(graph::Graph) = iterate(graph.statements)
Base.iterate(graph::Graph, state) = iterate(graph.statements, state)


# getmapping(graph::Graph, constant, default) = constant
function getmapping(graph::Graph, node)
    ref = graph.reference_mapping[node]
    return Reference(ref.number, getmutation(graph, ref))
end
function getmapping(graph::Graph, node, default)
    if haskey(graph.reference_mapping, node)
        getmapping(graph, node)
    else
        return default
    end
end
setmapping!(graph::Graph, (node, ref)::Pair) = graph.reference_mapping[node] = ref

function deletemapping!(graph::Graph, node)
    if haskey(graph.reference_mapping, node)
        delete!(graph, getmapping(graph, node))
    end
    return graph
end

getmutation(graph::Graph, ref::Reference) = get(graph.mutated_rvs, ref, ref.vn)
setmutation!(graph, (ref, vn)::Pair) = graph.mutated_rvs[ref] = vn
# function ismutated(graph, ref, ix)
#     if ref isa NamedReference
#         ref_sym = DynamicPPL.getsym(ref.vn)
#         ref_indexing = map.(ix -> try_getvalue(graph, ix), DynamicPPL.getindexing(ref.vn))

#         for mutated in values(graph.mutated_rvs)
#             if DynamicPPL.getsym(mutated) == ref_sym
#                 mut_indexing = map.(ix -> try_getvalue(graph, ix), DynamicPPL.getindexing(mutated))
#                 @show mut_indexing, ref_indexing
#                 DynamicPPL.subsumes(mut_indexing, ref_indexing) && return true
#             end
#         end
#     end
    
#     return false
# end


function makereference!(graph, node, vn=nothing)
    newref = Reference(length(graph.reference_mapping) + 1, vn)
    setmapping!(graph, node => newref)
    return newref
end

function Base.show(io::IO, graph::Graph)
    for (ref, stmt) in graph
        showstmt(io, ref, stmt)
    end
end

function showstmt(io::IO, ref, stmt)
    if stmt isa Assumption
        println(io, ref, (isdotted(stmt) ? " .~ " : " ~ "), stmt)
    elseif stmt isa Observation
        println(io, ref, (isdotted(stmt) ? " .⩪ " : " ⩪ "), stmt)
    else
        println(io, ref, " = ", stmt)
    end
end



try_getvalue(graph, ref::Reference) = getvalue(graph[ref])
try_getvalue(graph, constant) = constant

resolve_varname(graph, ref::UnnamedReference) = ref
function resolve_varname(graph, ref::NamedReference)
    var = ref.vn
    sym = DynamicPPL.getsym(var)
    indexing = DynamicPPL.getindexing(var)
    return VarName(sym, Tuple(try_getvalue.(Ref(graph), ix) for ix in indexing))
end


function convertcall(graph, node::CallingNode)
    f = convertvalue(graph, getfunction(node))
    args = convertvalue.(Ref(graph), getarguments(node))
    return Call(f, args, getvalue(node))
end

convertvalue(graph, expr::TapeReference) = getmapping(graph, expr[], getvalue(expr))
convertvalue(graph, expr::TapeConstant) = getvalue(expr)

convertdist!(graph, dist_expr::TapeConstant) = Constant(getvalue(dist))
function convertdist!(graph, dist_expr::TapeReference)
    ref = getmapping(graph, dist_expr[], nothing)
    if haskey(graph, ref)
        # move the separeate distribution call into the node and delete it from the graph
        dist = graph[ref]
        delete!(graph, ref)
        return dist
    else
        # the distribution node existed, but has already been deleted, so we reconstruct it
        # (obscure case when one distribution reference is sampled from twice)
        return convertcall(graph, dist_expr[])
    end
end


convertvn!(graph, vn_expr::Nothing) = nothing
convertvn!(graph, vn_expr::TapeConstant) = getvalue(vn_expr)
function convertvn!(graph, vn_expr::TapeReference)
    # @108: ⟨tuple⟩(@82) = (2,)
    # @109: ⟨tuple⟩(@108) = ((2,),)
    # @110: ⟨VarName⟩(⟨:x⟩, @109) = x[2]

    # extract the nodes that compose the indices of a varname
    vn_node = vn_expr[] # @110[]
    vn_name = getvalue(getargument(vn_node, 1))
    indices_node = try_getindex(getargument(vn_node, 2)) # @109[]
    index_element_nodes = getindex.(getarguments(indices_node)) # (@108[],)
    index_nodes = Tuple(try_getindex.(getarguments(index)) for index in index_element_nodes) # ((@82[],),)
    index_refs = map(ix -> getmapping.(Ref(graph), ix, ix), index_nodes)  # ((getmapping(graph, @82[]),),)

    # delete statements of all nodes that were involved in the construction (ie., @108, @109, @110)
    deletemapping!(graph, vn_node)
    deletemapping!(graph, indices_node)
    deletemapping!.(Ref(graph), index_element_nodes)

    return VarName(vn_name, index_refs)
end


function pushtilde!(graph, callingnode, maketilde)
    vn_expr, dist_expr, value_expr = tilde_parameters(callingnode)
    dist = convertdist!(graph, dist_expr)
    value = convertvalue(graph, value_expr)

    vn = convertvn!(graph, vn_expr)
    ref = makereference!(graph, callingnode, vn)
    graph[ref] = maketilde(dist, value)
    # setmutation!(graph, ref => vn)
    return graph
end


pushnode!(graph, node::CallingNode{typeof(DynamicPPL.tilde_assume)}) =
    pushtilde!(graph, node, Assumption{false})
pushnode!(graph, node::CallingNode{typeof(DynamicPPL.dot_tilde_assume)}) =
    pushtilde!(graph, node, Assumption{true})
pushnode!(graph, node::CallingNode{typeof(DynamicPPL.tilde_observe)}) =
    pushtilde!(graph, node, Observation{false})
pushnode!(graph, node::CallingNode{typeof(DynamicPPL.dot_tilde_observe)}) =
    pushtilde!(graph, node, Observation{true})

function pushnode!(graph, node::CallingNode)
    ref = makereference!(graph, node)
    graph[ref] = convertcall(graph, node)
    return graph
end

function pushnode!(graph, node::CallingNode{typeof(DynamicPPL.matchingvalue)})
    # special handling for model arguments
    # @7: ⟨getproperty⟩(@6, ⟨:x⟩) = [0.5, 1.1]                                                                                                                                                       
    # @8: ⟨DynamicPPL.matchingvalue⟩(@4, @3, @7) = [0.5, 1.1]

    value_expr = getargument(node, 3)
    if value_expr isa TapeReference
        getproperty_node = try_getindex(value_expr)
        delete!(graph, getmapping(graph, getproperty_node))
        argname = getvalue(getargument(getproperty_node, 2))
    else
        # this will probably never happen...?
        argname = gensym("argument")
    end

    ref = makereference!(graph, node)
    graph[ref] = Constant(getvalue(node))
    return graph
end

function pushnode!(graph, node::CallingNode{typeof(setindex!)})
    # @32: [§5:%34] ⟨DynamicPPL.tilde_assume⟩(@5, @4, @20, @29, @31, @3) = -0.031672055938280076
    # @33: [§5:%35] ⟨setindex!⟩(@19, @32, ⟨1⟩) = [...]
    argument_exprs = try_getindex.(getarguments(node))
    arguments = getmapping.(Ref(graph), argument_exprs, argument_exprs)
    mutated, value, indexing = arguments[1], arguments[2], arguments[3:end]
    if graph[value] isa Union{Assumption, Observation}
        setmutation!(graph, mutated => VarName(DynamicPPL.getsym(value.vn), (indexing,)))
    else
        invoke(pushnode!, Tuple{Graph, CallingNode}, graph, node)
    end

    
    return graph
end

function pushnode!(graph, node::CallingNode{typeof(getindex)})
    argument_exprs = try_getindex.(getarguments(node))
    arguments = getmapping.(Ref(graph), argument_exprs, argument_exprs)
    array, index = arguments[1], arguments[2:end]
    if array isa NamedReference
        ref = Reference(array.number, VarName(DynamicPPL.getsym(array.vn), (index,)))
        setmapping!(graph, node => ref)
        return graph
    else
        return invoke(pushnode!, Tuple{Graph, CallingNode}, graph, node)
    end
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


function eliminate_leftovers!(graph::Graph)
    marked = Set{Reference}()
    candidates = Set{Reference}()

    # first pass: mark everything between tildes
    for (ref, stmt) in graph
        empty!(candidates)
        
        if stmt isa Union{Assumption, Observation}
            # mark backwards from `ref`
            push!(candidates, ref)
            
            while !isempty(candidates)
                candidate = pop!(candidates)
                candidate ∈ marked && continue
                
                push!(marked, candidate)
                union!(candidates, dependencies(graph[candidate]))
                union!(candidates, dependencies(candidate))
            end
        end
        
    end

    # second pass: mark `setindex!` calls going to already marked refs
    for (ref, stmt) in graph
        if stmt isa Call{typeof(setindex!)}
            union!(candidates, dependencies(stmt))
            
            while !isempty(candidates)
                candidate = pop!(candidates)
                
                if candidate ∈ marked
                    push!(marked, ref)
                    break
                end
                
                union!(candidates, dependencies(graph[candidate]))
                union!(candidates, dependencies(candidate))
            end
        end
    end

    # third step: delete all unmarked nodes
    for unmarked in setdiff(keys(graph), marked)
        delete!(graph, unmarked)
    end
end

function makegraph(slice::Vector{<:AbstractNode})
    graph = foldl(pushnode!, slice, init=Graph())
    eliminate_leftovers!(graph)
    return graph
end


function trackdependencies(model)
    trace = trackmodel(model)
    dependency_slice = strip_dependencies(strip_model_layers(trace))
    return makegraph(dependency_slice)
end





