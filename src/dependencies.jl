using DynamicPPL
using Distributions

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
            if getvalue(child.call.arguments[2]) == :args && try_getindex(child.call.arguments[1]) == model_node
                push!(modelargs_nodes, child)
            elseif try_getindex(child.call.arguments[1]) ∈ modelargs_nodes
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

struct Assumption{T, TDist<:Statement} <: Statement
    dist::TDist
    value::T
end

struct Observation{T, TDist<:Statement} <: Statement
    dist::TDist
    value::T
end

struct Call{T, TF, TArgs<:Tuple} <: Statement
    f::TF
    args::TArgs
    value::T
end

struct Constant{T} <: Statement
    value::T
end

struct Argument{T} <: Statement
    name::String
    value::T
end


shortname(d::Type{<:Distribution}) = string(nameof(d))
shortname(other) = string(other)

Base.show(io::IO, stmt::Assumption) =
    print(io, shortname(stmt.dist.f), "(", join(stmt.dist.args, ", "), ") → ", stmt.value)
Base.show(io::IO, stmt::Observation) =
    print(io, shortname(stmt.dist.f), "(", join(stmt.dist.args, ", "), ") → ", stmt.value)
Base.show(io::IO, stmt::Call) = print(io, stmt.f, "(", join(stmt.args, ", "), ") → ", stmt.value)
Base.show(io::IO, stmt::Constant) = print(io, stmt.value)
Base.show(io::IO, stmt::Argument) = print(io, stmt.name, " → ", stmt.value)

AutoGibbs.getvalue(stmt::Statement) = stmt.value


struct Reference{TV<:Union{VarName, Nothing}}
    number::Int
    vn::TV
end

Reference(number) = Reference(number, nothing)


Base.show(io::IO, r::Reference{Nothing}) = print(io, "%", r.number)
Base.show(io::IO, r::Reference{<:VarName}) = print(io, "%", r.number, ":", r.vn)
Base.isless(q::Reference, r::Reference) = isless(q.number, r.number)
Base.hash(r::Reference, h::UInt) = hash(r.number, h)
Base.:(==)(q::Reference, r::Reference) = q.number == r.number


struct Graph
    statements::Dict{Reference, Statement}
    reference_mapping::Dict{AbstractNode, Reference}
    mutated_rvs::Dict{Reference, VarName}
end

Graph() = Graph(Dict{Reference, Statement}(),
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


getmapping(graph::Graph, constant) = constant
getmapping(graph::Graph, node::AbstractNode) = get(graph.reference_mapping, node, node)
setmapping!(graph::Graph, (node, ref)::Pair) = graph.reference_mapping[node] = ref

getmutation(graph::Graph, ref::Reference) = get(graph.mutated_rvs, ref, ref.vn)
setmutation!(graph, (ref, vn)::Pair) = graph.mutated_rvs[ref] = vn


function makereference!(graph, node, vn=nothing)
    newref = Reference(length(graph.reference_mapping) + 1, vn)
    setmapping!(graph, node => newref)
    return newref
end

function Base.show(io::IO, graph::Graph)
    for (ref, stmt) in graph
        if stmt isa Assumption
            println(io, ref, " ~ ", stmt)
        elseif stmt isa Observation
            println(io, ref, " ⩪ ", stmt)
        else
            println(io, ref, " = ", stmt)
        end
    end
end



try_getvalue(graph, ref::Reference) = getvalue(graph[ref])
try_getvalue(graph, constant) = constant

function resolve_varname(graph, ref::Reference{<:VarName})
    var = ref.vn
    sym = DynamicPPL.getsym(var)
    indexing = DynamicPPL.getindexing(var)
    return VarName(sym, Tuple(try_getvalue.(Ref(graph), ix) for ix in indexing))
end


function makecallnode(graph, node::CallingNode)
    f = convertvalue(graph, node.call.f)
    args = convertvalue.(Ref(graph), (node.call.arguments..., something(node.call.varargs, ())...))
    return Call(f, args, getvalue(node))
end

convertvalue(graph, expr::TapeReference) = get(graph.reference_mapping, expr[], getvalue(expr))
convertvalue(graph, expr::TapeConstant) = getvalue(expr)

convertdist(graph, dist_expr::TapeConstant) = Constant(getvalue(dist))
function convertdist(graph, dist_expr::TapeReference)
    ref = getmapping(graph, dist_expr[])
    if haskey(graph, ref)
        # move the separeate distribution call into the node and delete it from the graph
        dist = graph[ref]
        delete!(graph, ref)
        return dist
    else
        # the distribution node existed, but has already been deleted, so we reconstruct it
        # (obscure case when one distribution reference is sampled from twice)
        return makecallnode(graph, dist_expr[])
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
    vn_name = getvalue(vn_node.call.arguments[1])
    indices_node = try_getindex(vn_node.call.arguments[2]) # @109[]
    index_element_nodes = getindex.(indices_node.call.arguments) # (@108[],)
    index_nodes = Tuple(try_getindex.(index.call.arguments) for index in index_element_nodes) # ((@82[],),)
    index_refs = map(ix -> getmapping.(Ref(graph), ix), index_nodes)  # ((getmapping(graph, @82[]),),)

    # delete all nodes that were involved in the construction (ie., @108, @109, @110)
    delete!(graph, getmapping(graph, vn_node))
    delete!(graph, getmapping(graph, indices_node))
    delete!.(Ref(graph), getmapping.(Ref(graph), index_element_nodes))

    return VarName(vn_name, index_refs)
end

function pushtilde!(graph, callingnode, maketilde)
    vn_expr, dist_expr, value_expr = tilde_parameters(callingnode)
    dist = convertdist(graph, dist_expr)
    value = convertvalue(graph, value_expr)

    vn = convertvn!(graph, vn_expr)
    ref = makereference!(graph, callingnode, vn)
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

    value_expr = node.call.arguments[3]
    if value_expr isa TapeReference
        getproperty_node = try_getindex(value_expr)
        delete!(graph, getmapping(graph, getproperty_node))
        argname = getvalue(getproperty_node.call.arguments[2])
    else
        argname = gensym("argument")
    end

    ref = makereference!(graph, node)
    graph[ref] = Argument(string(argname), getvalue(node))
    return graph
end

function pushnode!(graph, node::CallingNode{typeof(setindex!)})
    # @32: [§5:%34] ⟨DynamicPPL.tilde_assume⟩(@5, @4, @20, @29, @31, @3) = -0.031672055938280076
    # @33: [§5:%35] ⟨setindex!⟩(@19, @32, ⟨1⟩) = [...]
    ref = makereference!(graph, node)
    graph[ref] = makecallnode(graph, node)

    mutated, value = graph[ref].args[1], graph[ref].args[2]
    if graph[value] isa Union{Assumption, Observation}
        setmutation!(graph, mutated => VarName(DynamicPPL.getsym(value.vn)))
    end
    
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


function replace_mutated!(graph)
    mutated_rvs = graph.mutated_rvs

    for (ref, vn) in mutated_rvs
        original_stmt = graph[ref]
        delete!(graph, ref)
        new_ref = Reference(ref.number, vn)
        graph[new_ref] = original_stmt
    end
    
    for (ref, stmt) in graph
        graph[ref] = replace_mutated(graph, graph[ref])
    end

    return graph
end

replace_mutated(graph, ref::Reference) = Reference(ref.number, getmutation(graph, ref))
replace_mutated(graph, stmt::Assumption) = Assumption(replace_mutated(graph, stmt.dist), stmt.value)
replace_mutated(graph, stmt::Observation) = Observation(replace_mutated(graph, stmt.dist), stmt.value)
replace_mutated(graph, stmt::Call) = Call(replace_mutated(graph, stmt.f),
                                          replace_mutated.(Ref(graph), stmt.args),
                                          stmt.value)
replace_mutated(graph, stmt::Constant) = stmt
replace_mutated(graph, stmt::Argument) = stmt
replace_mutated(graph, constant) = constant

function makegraph(slice::Vector{<:AbstractNode})
    return replace_mutated!(foldl(pushnode!, slice, init=Graph()))
end


function trackdependencies(model)
    trace = trackmodel(model)
    dependency_slice = strip_dependencies(strip_model_layers(trace))
    return makegraph(dependency_slice)
end





