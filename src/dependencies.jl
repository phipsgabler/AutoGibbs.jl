using IRTracker
using DynamicPPL
using Distributions
using DataStructures: SortedDict

export strip_dependencies, makegraph, showgraph


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
    args = getarguments(node) # rng, ctx, sampler, right, vn, inds, vi
    vn, dist, value, vi = args[5], args[4], getvalue(node), args[7]
    return vn, dist, TapeConstant(value), vi
end

function tilde_parameters(node::CallingNode{typeof(DynamicPPL.tilde_observe)})
    args = getarguments(node)
    if length(args) == 7
        # ctx, sampler, right, left, vname, vinds, vi
        vn, dist, value, vi = args[5], args[3], args[4], args[7]
        return vn, dist, value, vi
    elseif length(args) == 5
        # ctx, sampler, right, left, vi
        dist, value, vi = args[3], args[4], args[5]
        return nothing, dist, value, vi
    else
        throw(ArgumentError("$node has unknown argument structure, you should not have reached this point!"))
    end
end

function tilde_parameters(node::CallingNode{typeof(DynamicPPL.dot_tilde_assume)})
    args = getarguments(node) # rng, ctx, sampler, right, left, vn, inds, vi
    vn, dist, value, vi = args[6], args[4], args[5], args[8]
    return vn, dist, value, vi
end

function tilde_parameters(node::CallingNode{typeof(DynamicPPL.dot_tilde_observe)})
    args = getarguments(node)
    if length(args) == 7
        # ctx, sampler, right, left, vn, inds, vi
        vn, dist, value, vi = args[5], args[3], args[4], args[7]
        return vn, dist, value, vi
    elseif length(args) == 5
        # ctx, sampler, right, left, vi
        dist, value, vi = args[3], args[4], args[5]
        return nothing, dist, value, vi
    else
        throw(ArgumentError("$node has unknown argument structure, you should not have reached this point!"))
    end
end


"""
    model_argument_nodes(root)

Extract from a stripped model trace all model argument `getindex` calls.
"""
function model_argument_nodes(root)
    # From the beginning of the trace,
    # ```
    # @1: [Arg:§1:%1] @11#1 → ##evaluator#465
    # @2: [Arg:§1:%2] @11#2 → Random._GLOBAL_RNG()
    # @3: [Arg:§1:%3] @11#3 → Model{...}
    # @4: [Arg:§1:%4] @11#4 → VarInfo (0 variables, dimension 0; logp: 0.0)
    # @5: [Arg:§1:%5] @11#5 → SampleFromPrior()
    # @6: [Arg:§1:%6] @11#6 → DefaultContext()
    # @7: [§1:%7] ⟨getproperty⟩(@3, ⟨:args⟩) → (x = [0.1, 0.05, 1.0],)
    # @8: [§1:%8] ⟨getproperty⟩(@7, ⟨:x⟩) → [0.1, 0.05, 1.0]
    # ```
    # extract only the `getproperty(@n, ⟨:varname⟩)` line for each argument.

    argument_nodes = Vector{AbstractNode}()
    model_node = getchildren(root)[3]
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




struct Reference
    number::Int
end

Base.show(io::IO, r::Reference) = print(io, "⟨", r.number, "⟩")
Base.isless(q::Reference, r::Reference) = isless(q.number, r.number)
Base.hash(r::Reference, h::UInt) = hash(r.number, h)
Base.:(==)(q::Reference, r::Reference) = q.number == r.number


abstract type Statement end

struct Assumption{dotted, TN<:VarName, TDist<:Statement, TVal} <: Statement
    vn::TN
    dist::TDist
    value::TVal
    logp::Float64
end

Assumption{dotted}(vn, dist, value, logp) where {dotted} =
    Assumption{dotted, typeof(vn), typeof(dist), typeof(value)}(vn, dist, value, logp)

struct Observation{dotted, TN<:Union{Nothing, VarName}, TDist<:Statement, TVal} <: Statement
    vn::TN
    dist::TDist
    value::TVal
    logp::Float64
end

Observation{dotted}(vn, dist, value, logp) where {dotted} =
    Observation{dotted, typeof(vn), typeof(dist), typeof(value)}(vn, dist, value, logp)

struct Call{TD<:Union{Nothing, Tuple{VarName, Reference}}, TF, TArgs<:Tuple, TVal} <: Statement
    definition::TD
    f::TF
    args::TArgs
    value::TVal
end

Call(f, args, value) = Call(nothing, f, args, value)

struct Constant{TVal} <: Statement
    value::TVal
end


const Tilde{dotted} = Union{Assumption{dotted}, Observation{dotted}}



_shortname(d::Type{<:Distribution}) = string(nameof(d))
_shortname(other) = string(other)

function Base.show(io::IO, stmt::Assumption)
    print(io, stmt.vn, (isdotted(stmt) ? " .~ " : " ~ "), _shortname(stmt.dist.f), "(")
    join(io, stmt.dist.args, ", ")
    print(io, ") → ", stmt.value)
end
function Base.show(io::IO, stmt::Observation)
    print(io, (isnothing(stmt.vn) ? stmt.value : stmt.vn))
    print(io, (isdotted(stmt) ? " .⩪ " : " ⩪ "), _shortname(stmt.dist.f), "(")
    join(io, stmt.dist.args, ", ")
    print(io, ") ← ", stmt.value)
end
function Base.show(io::IO, stmt::Call)
    if !isnothing(stmt.definition)
        vn = stmt.definition[1]
        print(io, vn, " = ")
    end
    print(io, stmt.f)
    print(io, "(")
    join(io, stmt.args, ", ")
    print(io, ") → ", stmt.value)
end
Base.show(io::IO, stmt::Constant) = print(io, stmt.value)

IRTracker.getvalue(stmt::Statement) = stmt.value

isdotted(::Type{<:Assumption{dotted}}) where {dotted} = dotted
isdotted(::Type{<:Observation{dotted}}) where {dotted} = dotted
isdotted(tilde::Tilde) = isdotted(typeof(tilde))


getvn(stmt::Tilde) = stmt.vn
getvn(stmt::Call) = isnothing(stmt.definition) ? nothing : stmt.definition[1]
getvn(::Constant) = nothing


"""
    parent_variables(graph, stmt)

Return all `Assumption`s (potentially with indexing) that the tilde `stmt` depends on directly.
"""
function parent_variables(graph, stmt::Tilde)
    result = Set{Tuple{Assumption, Union{Tuple, Nothing}}}()
    parent_variables!(result, graph, stmt.dist)
    stmt.value isa Reference && parent_variables!(result, graph, stmt.value)
    return result
end

parent_variables!(result, graph, stmt) = result
function parent_variables!(result, graph, ref::Reference)
    # follow back implicit dependencies on unmarked locations without indexing, e.g.
    # ⟨2⟩ = [0.1, -0.2]
    # ⟨3⟩ = Array{Float64,1}(array initializer with undefined values, 2) → ...
    # ⟨6⟩ = m[1] ~ Normal() → ...
    # ⟨7⟩ = m[1] = getindex(⟨3⟩, 1) → ...
    # ⟨10⟩ = m[2] ~ Normal() → ...
    # ⟨11⟩ = m[2] = getindex(⟨3⟩, 2) → ...
    # ⟨13⟩ = x ⩪ MvNormal(⟨3⟩) ← ⟨2⟩
    # `x` depends on the whole of `m`, stored in `⟨3⟩`, and thus implicitly
    # on `m[1]` and `m[2]` (cf. reverse_deps test case).
    
    if haskey(graph.rv_locations, ref)
        for (ix, loc) in graph.rv_locations[ref]
            loc == ref && continue # skip one-line array tildes, e.g., z ~ filldist(...)
            parent_variables!(result, graph, loc)
        end
    end
    
    return parent_variables!(result, graph, graph[ref])
end
function parent_variables!(result, graph, stmt::Assumption)
    # direct dependency:
    # ⟨31⟩ = w ~ Dirichlet()
    # ⟨42⟩ = z[1] ~ DiscreteNonParametric(⟨31⟩)
    return push!(result, (stmt, nothing))
end
function parent_variables!(result, graph, stmt::Observation)
    @warn "The parent statement is an observation ($(stmt)), something weird has happened..."
    return result
end
function parent_variables!(result, graph, stmt::Call{<:Nothing})
    # go through all argumens of an unmarked location:
    # ⟨85⟩ = getindex(⟨2⟩, ⟨72⟩) → 1.0
    # ⟨86⟩ = x[3] ⩪ Normal(⟨80⟩, 1.0) ← ⟨85⟩
    
    for arg in stmt.args
        parent_variables!(result, graph, arg)
    end
    
    return result
end
function parent_variables!(result, graph, stmt::Call{<:Tuple})
    # also go back to variable definition site of a marked location:
    # ⟨28⟩ = μ[2] ~ Normal() → 0.851814791967429
    # ⟨80⟩ = μ[2] = getindex(⟨9⟩, ⟨79⟩) → 0.851814791967429
    # ⟨86⟩ = x[3] ⩪ Normal(⟨80⟩, 1.0) ← ⟨85⟩

    for arg in stmt.args
        parent_variables!(result, graph, arg)
    end
    vn, location = stmt.definition
    push!(result, (graph[location], DynamicPPL.getindexing(vn)))
    return result
end



struct Graph
    """Entries in graph: mapping from references to statements."""
    statements::SortedDict{Reference, Statement}

    """Association which nodes in the trace are associated with which references in the graph"""
    reference_mapping::Dict{AbstractNode, Reference}

    """
    Remembers in which references random variables are actually stored; this is either the location
    of the tilde statement (`beta ~ Normal()`, `z ~ filldist(...)`), or the array position it is
    assigned to (`z[i] ~ Normal()`).

    The container is first ordered by reference of storage, then by index for `subsumes` testing.
    """
    rv_locations::Dict{Reference, Vector{Pair{Tuple, Reference}}}
end

Graph() = Graph(SortedDict{Reference, Statement}(),
                Dict{AbstractNode, Reference}(),
                Dict{Reference, Vector{Pair{Tuple, Reference}}}())

Base.IteratorSize(::Type{Graph}) = Base.HasLength()
Base.length(graph::Graph) = length(graph.statements)
Base.IteratorEltype(::Type{Graph}) = Base.HasEltype()
Base.eltype(graph::Graph) = eltype(graph.statements)
Base.getindex(graph::Graph, ref) = graph.statements[ref]
Base.setindex!(graph::Graph, stmt, ref) = graph.statements[ref] = stmt
Base.firstindex(graph::Graph) = firstindex(graph.statements)
Base.lastindex(graph::Graph) = lastindex(graph.statements)
Base.get(graph::Graph, ref, default) = get(graph.statements, ref, default)
Base.haskey(graph::Graph, ref) = haskey(graph.statements, ref)
Base.delete!(graph::Graph, ref) = delete!(graph.statements, ref)
Base.keys(graph::Graph) = keys(graph.statements)
Base.values(graph::Graph) = values(graph.statements)
Base.iterate(graph::Graph) = iterate(graph.statements)
Base.iterate(graph::Graph, state) = iterate(graph.statements, state)

# just intended for debugging
Base.getindex(graph::Graph, i::Int) = graph[Reference(i)]

# to convert an atom to a runtime value: either the value of the referenced node, or a literal
tovalue(graph, ref::Reference) = getvalue(graph[ref])
tovalue(graph, other) = other

function Base.mapreduce(f, op, graph::Graph; init)
    for kv in graph
        init = op(init, f(kv))
    end

    return init
end

function Base.filter(f, graph::Graph; init=Vector{eltype(graph)}())
    for kv in graph
        f(kv) && push!(init, kv)
    end

    return init
end

Base.map(f, graph::Graph; init=Vector{eltype(graph)}()) = mapreduce(f, push!; init=init)


getstatements(graph::Graph) = graph.statements

function getmapping(graph::Graph, node)
    ref = graph.reference_mapping[node]
    return ref
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


"""Look up the matching location of an RV in graph.rv_locations[ref] for given index tuple."""
function _find_indexing(indexings, indexing)
    for (ix, r) in indexings
        if DynamicPPL.subsumes(ix, indexing)
            return r
        end
    end
    return nothing
end

function getrv(graph::Graph, ref::Reference, indexing)
    _find_indexing(graph.rv_locations[ref], indexing)
end

function setrv!(graph, ((ref, indexing), vn))
    push!(get!(graph.rv_locations, ref, valtype(graph.rv_locations)()),
          indexing => vn)
end

function hasrv(graph, ref, indexing)
    indexings = get(graph.rv_locations, ref, nothing)
    return !isnothing(indexings) && !isnothing(_find_indexing(indexings, indexing))
end


function makereference!(graph, node)
    newref = Reference(length(graph.reference_mapping) + 1)
    setmapping!(graph, node => newref)
    return newref
end

function Base.show(io::IO, graph::Graph)
    for (ref, stmt) in graph
        println(io, ref, " = ", stmt)
    end
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
        # move the separate distribution call into the node and delete it from the graph
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

    return VarName(vn_name, map.(ix -> tovalue(graph, ix), index_refs))
end


function pushtilde!(graph, callingnode, tilde_constructor)
    vn_expr, dist_expr, value_expr, vi_expr = tilde_parameters(callingnode)
    vn = convertvn!(graph, vn_expr)
    dist = convertdist!(graph, dist_expr)
    value = convertvalue(graph, value_expr)

    ref = makereference!(graph, callingnode)
    graph[ref] = tilde_constructor(vn, dist, value, -Inf)

    if !isnothing(vn)
        indexing_values = tovalue.(Ref(graph), DynamicPPL.getindexing(vn))
        if isdotted(tilde_constructor)
            @warn "Broadcasted tildes ($(graph[ref])) are not fully supported!"
            setrv!(graph, (value, indexing_values) => ref)
        elseif indexing_values == ()
            # this is the case where an unindexed variable is sampled -- the value is equivalent to
            # the current reference; otherwise, a `setindex!` should follow in the next line.
            setrv!(graph, (ref, indexing_values) => ref)
        end
    end
    
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

    value_expr = getargument(node, 3) # @7 above
    if value_expr isa TapeReference
        getproperty_node = value_expr[]
        delete!(graph, getmapping(graph, getproperty_node))
    else # value vas a literal, not a function argument
        @warn "this probably shouldn't have happened..."
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
    indexing_values = (tovalue.(Ref(graph), indexing),)
    
    if value isa Reference && graph[value] isa Tilde
        # this is the case where in the previous line, an indexed RV was sampled; we need
        # to associate the value with the array position it is assigned to here.
        setrv!(graph, (mutated, indexing_values) => value)
        
        # # we can also replace the `setindex!` call that followed the tilde node
        # # by a `getindex` on the same value, to preserve the information of the dependency of
        # # between the variable, the array, and the index
        # # PROBABLY UNNEEDED? see reverse_deps test case in test_conditionals
        definition = (graph[value].vn, value)
        ref = makereference!(graph, node)
        graph[ref] = Call(definition, getindex, (mutated, indexing...), tovalue(graph, value))
    else
        invoke(pushnode!, Tuple{Graph, CallingNode}, graph, node)
    end
    
    return graph
end

function pushnode!(graph, node::CallingNode{typeof(getindex)})
    # @180: ⟨getindex⟩(@9, @135) → -0.05
    argument_exprs = try_getindex.(getarguments(node))
    arguments = getmapping.(Ref(graph), argument_exprs, argument_exprs)
    array, indexing = arguments[1], arguments[2:end]
    indexing_values = (tovalue.(Ref(graph), indexing),)
    
    if hasrv(graph, array, indexing_values)
        rv = getrv(graph, array, indexing_values)
        ref = makereference!(graph, node)
        definition = (VarName(graph[rv].vn, indexing_values), rv)
        graph[ref] = Call(definition, getindex, (array, indexing...), getvalue(node))
    else
        invoke(pushnode!, Tuple{Graph, CallingNode}, graph, node)
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
        # don't add, but remember branch arguments
        original_ref = getmapping(graph, referenced(node)[1])
        setmapping!(graph, node => original_ref)
    else
        # handle function argument nodes like constants
        ref = makereference!(graph, node)
        graph[ref] = Constant(getvalue(node))
    end

    return graph
end


function makegraph(slice::Vector{<:AbstractNode})
    return foldl(slice, init=Graph()) do graph, node
        pushnode!(graph, node)
    end
end




