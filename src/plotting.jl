using IRTracker.GraphViz

deps(::Constant) = Reference[]
deps(node::Assumption) = deps(node.dist)
deps(node::Observation) = deps(node.dist)
deps(node::Call) = [arg for arg in node.args if arg isa Reference]


function pushstmt!(stmts, ref, stmt::Constant)
    value = escape_string(sprint(showstmt, ref, stmt))
    push!(stmts, GraphViz.Node(string(ref.number), label=value, shape="rectangle"))
end
function pushstmt!(stmts, ref, stmt::Call)
    argstring = join(stmt.args, ", ")
    label = escape_string(sprint(showstmt, ref, stmt))
    push!(stmts, GraphViz.Node(string(ref.number), label=label, shape="rectangle"))
end
function pushstmt!(stmts, ref, stmt::Union{Assumption, Observation})
    label = escape_string(sprint(showstmt, ref, stmt))
    push!(stmts, GraphViz.Node(string(ref.number), label=label, shape="circle"))
end


function Base.convert(::Type{GraphViz.Graph}, graph::Graph)
    stmts = Vector{GraphViz.Statement}()

    for (ref, stmt) in graph
        pushstmt!(stmts, ref, stmt)
        
        for dep in deps(stmt)
            from = GraphViz.NodeID(string(ref.number))
            to = GraphViz.NodeID(string(dep.number))
            push!(stmts, GraphViz.Edge([from, to]))
        end

        if ref isa Reference{<:VarName}
            from = GraphViz.NodeID(string(ref.number))
            @show ref
            for index in DynamicPPL.getindexing(ref.vn), ix in index
                if ix isa Reference
                    to = GraphViz.NodeID(string(ix.number))
                    @show ix
                    push!(stmts, GraphViz.Edge([from, to]))
                end
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


# dot /tmp/graph.dot -Tpdf -Nfontname="DejaVu Sans Mono" -Efontname="DejaVu Sans Mono" > /tmp/graph.pdf

