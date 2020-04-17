using IRTracker.GraphViz

deps(::Constant) = Reference[]
deps(node::Assumption) = [node.dist]
deps(node::Observation) = [node.dist]
deps(node::Call) = [arg for arg in node.args if arg isa Reference]


function pushnode!(stmts, r, node::Constant)
    value = escape_string(string(node.value))
    push!(stmts, GraphViz.Node(string(r.name), label=value, shape="rectangle"))
end
function pushnode!(stmts, r, node::Call)
    argstring = join(node.args, ", ")
    label = escape_string("$r = $(node.f)($argstring)")
    push!(stmts, GraphViz.Node(string(r.name), label=label, shape="rectangle"))
end
function pushnode!(stmts, r, node::Union{Assumption, Observation})
    label = escape_string("$r = $(node.vn) ~ $(node.dist)")
    push!(stmts, GraphViz.Node(string(r.name), label=label, shape="circle"))
end


function convert(::Type{GraphViz.Graph}, graph::Dict{Reference, DepNode})
    stmts = Vector{GraphViz.Statement}()

    for (r, node) in graph
        pushnode!(stmts, r, node)
        
        for dep in deps(node)
            from = GraphViz.NodeID(string(r.name))
            to = GraphViz.NodeID(string(dep.name))
            push!(stmts, GraphViz.Edge([from, to]))
        end
    end
    
    graph_attrs = Dict(:ordering => "in",    # edges sorted by incoming
                       :rankdir => "BT",     # order nodes from right to left
                       )
    edge_attrs = Dict{Symbol, String}()
    node_attrs = Dict(:shape => "plaintext")
    
    return GraphViz.DiGraph(stmts,
                            graph_attrs = graph_attrs,
                            edge_attrs = edge_attrs,
                            node_attrs = node_attrs)
end


function savedot(fn::AbstractString, graph::Dict{Reference, DepNode})
    open(fn, "w") do fp
        GraphViz.pprint(fp, convert(GraphViz.Graph, graph))
    end
end


# dot /tmp/graph.dot -Tpdf -Nfontname="DejaVu Sans Mono" -Efontname="DejaVu Sans Mono" > /tmp/graph.pdf

