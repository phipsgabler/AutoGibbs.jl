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



macro test_nothrow(ex)
    orig_ex = Expr(:inert, ex)
    truthy_ex = quote
        $(esc(ex))
        true
    end
    result = quote
        try
            Test.Returned($truthy_ex, nothing, $(QuoteNode(__source__)))
        catch _e
            _e isa Test.InterruptException && rethrow()
            Test.Returned(false, nothing, $(QuoteNode(__source__)))
        end
    end
    Base.remove_linenums!(result)
    :(Test.do_test($result, $orig_ex))
end

function issimilar(d1::DiscreteNonParametric, d2::DiscreteNonParametric; atol::Real=0)
    return all(isapprox.(support(d1), support(d2); atol=atol)) &&
        all(isapprox.(probs(d1), probs(d2); atol=atol))
end
