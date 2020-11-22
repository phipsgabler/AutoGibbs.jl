using IRTracker.GraphViz

escape_show(x) = escape_string(sprint(show, x))

function pushstmt!(stmts, ref, stmt::Constant)
    value = escape_show(ref) * " = " * escape_show(stmt.value)
    push!(stmts, GraphViz.Node(string(ref.number), label=value, shape="rectangle"))
end
function pushstmt!(stmts, ref, stmt::Call)
    argstring = join((escape_show(arg) for arg in stmt.args), ", ")
    label = escape_show(ref) * " = " * escape_show(stmt.f) * "(" * argstring * ")"
    push!(stmts, GraphViz.Node(string(ref.number), label=label, shape="rectangle"))
end
function pushstmt!(stmts, ref, stmt::Assumption)
    label = "$(escape_show(ref)) = $(escape_show(stmt.vn)) ~ $(escape_show(stmt.dist_ref))"
    push!(stmts, GraphViz.Node(string(ref.number), label=label, shape="circle"))
end
function pushstmt!(stmts, ref, stmt::Observation)
    label = "$(escape_show(ref)) = $(escape_show(stmt.vn)) ~̇ $(escape_show(stmt.dist_ref)) ← $(escape_show(stmt.value))"
    push!(stmts, GraphViz.Node(string(ref.number), label=label, shape="circle"))
end


function Base.convert(::Type{GraphViz.Graph}, graph::Graph)
    stmts = Vector{GraphViz.Statement}()
    nodes = Dict{Statement, GraphViz.NodeID}()

    for (ref, stmt) in graph
        # node for stmt itself
        pushstmt!(stmts, ref, stmt)
        from = GraphViz.NodeID(string(ref.number))
        nodes[stmt] = from

        if stmt isa Tilde
            # tilde to distribution
            dist = nodes[graph[stmt.dist_ref]]
            push!(stmts, GraphViz.Edge([from, dist]))

            # tilde to observed value
            if stmt isa Observation && stmt.value isa Reference
                to = nodes[graph[stmt.value]]
                push!(stmts, GraphViz.Edge([from, to]))
            end
        elseif stmt isa Call
            # call to each parent argument
            for arg in stmt.args
                if arg isa Reference
                    to = nodes[graph[arg]]
                    push!(stmts, GraphViz.Edge([from, to]))
                end
            end

            # call to non-constant function
            if stmt.f isa Reference
                to = nodes[graph[stmt.f]]
                push!(stmts, GraphViz.Edge([from, to]))
            end
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
        GraphViz.pprint(fp, Base.convert(GraphViz.Graph, graph))
    end
end


# dot /tmp/graph.dot -Tpdf -Nfontname="Noto Mono" -Efontname="Noto Mono" > /tmp/graph.pdf

