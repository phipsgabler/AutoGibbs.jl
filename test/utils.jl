function varnames(graph)
    vars = (ref for ref in keys(graph) if ref isa AutoGibbs.NamedReference)
    return Set(AutoGibbs.resolve_varname(graph, ref) for ref in vars)
end


"""Check whether all dependencies of all variables in a graph in a graph are captured."""
function check_dependencies(graph)
    refs = keys(graph)
    subsumes(q, r) = isnothing(q) || (!isnothing(r) && DynamicPPL.subsumes(q, r))
    matches(dep) = any(ref.number == dep.number && subsumes(ref.vn, dep.vn) for ref in keys(graph))
    return all(matches(dep) for expr in values(graph) for dep in AutoGibbs.dependencies(expr))
end


track_with_context(model, ctx) = trackdependencies(model, VarInfo(), SampleFromPrior(), ctx)


macro test_dependency_invariants(graph, vns)
    return quote
        @test varnames($(esc(graph))) == Set{VarName}($(esc(vns)))
        @test check_dependencies($(esc(graph)))
    end
end


macro testdependencies(model, varname_exprs...)
    return quote
        let m = $(esc(model)),
            vns = tuple($(DynamicPPL.varname.(varname_exprs)...))
            
            graph_default = trackdependencies(m)
            @test_dependency_invariants(graph_default, vns)
        
            graph_likelihood = track_with_context(m, LikelihoodContext())
            @test_dependency_invariants(graph_likelihood, vns)
        end
    end
end
