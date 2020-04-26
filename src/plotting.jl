using IRTracker.GraphViz

deps(::Constant) = Reference[]
deps(::Argument) = Reference[]
deps(node::Assumption) = deps(node.dist)
deps(node::Observation) = deps(node.dist)
deps(node::Call) = [arg for arg in node.args if arg isa Reference]


function pushnode!(stmts, r, stmt::Constant)
    value = escape_string(string(stmt.value))
    push!(stmts, GraphViz.Node(string(r.name), label=value, shape="rectangle"))
end
function pushnode!(stmts, r, stmt::Call)
    argstring = join(stmt.args, ", ")
    label = escape_string("$r = $(stmt.f)($argstring)")
    push!(stmts, GraphViz.Node(string(r.name), label=label, shape="rectangle"))
end
function pushnode!(stmts, r, stmt::Union{Assumption, Observation})
    dist_args = join(stmt.dist.args, ", ")
    label = escape_string("$r = $(stmt.vn) ~ $(stmt.dist.f)($dist_args)")
    push!(stmts, GraphViz.Node(string(r.name), label=label, shape="circle"))
end
function pushnode!(stmts, r, stmt::Argument)
    label = escape_string("$r = $(stmt.name) = $(stmt.value)")
    push!(stmts, GraphViz.Node(string(r.name), label=label, shape="rectangle"))
end


function convert(::Type{GraphViz.Graph}, graph::Graph)
    stmts = Vector{GraphViz.Statement}()

    for (r, stmt) in graph
        pushnode!(stmts, r, stmt)
        
        for dep in deps(stmt)
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


function savedot(fn::AbstractString, graph::Graph)
    open(fn, "w") do fp
        GraphViz.pprint(fp, convert(GraphViz.Graph, graph))
    end
end


# dot /tmp/graph.dot -Tpdf -Nfontname="DejaVu Sans Mono" -Efontname="DejaVu Sans Mono" > /tmp/graph.pdf

