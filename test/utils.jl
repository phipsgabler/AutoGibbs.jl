

function varnames(graph)
    result = Set{VarName}()
    
    for stmt in values(graph)
        if stmt isa AutoGibbs.Tilde
            vn = AutoGibbs.getvn(stmt)
            if !isnothing(vn)
                push!(result, vn)
            end
        end
    end

    return result
end


track_with_context(model, ctx) = trackdependencies(model, VarInfo(), SampleFromPrior(), ctx)


macro testdependencies(model, varname_exprs...)
    result = quote
        let m = $(esc(model)),
            vns = tuple($(DynamicPPL.varname.(varname_exprs)...))
            
            graph_default = trackdependencies(m)
            @test varnames(graph_default) ⊇ Set{VarName}(vns)
            # @test check_dependencies(graph_default)
        
            # graph_likelihood = track_with_context(m, LikelihoodContext())
            # $(test_dependency_invariants(:graph_likelihood))
        end
    end

    Base.remove_linenums!(result)
    return result
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
    return :(Test.do_test($result, $orig_ex))
end

function issimilar(d1::DiscreteNonParametric, d2::DiscreteNonParametric; atol::Real=0)
    return (support(d1) == support(d2)) &&
        all(isapprox.(probs(d1), probs(d2); atol=atol))
end

function issimilar(p1::Product, p2::Product; atol::Real=0)
    return all(issimilar(d1, d2) for (d1, d2) in zip(p1.v, p2.v))
end
