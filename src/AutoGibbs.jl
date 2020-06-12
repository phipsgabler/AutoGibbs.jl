module AutoGibbs

include("tracking.jl")
include("dependencies.jl")
include("plotting.jl")
# include("vartrie.jl")
include("conditionals.jl")
include("auto_conditional.jl")


export trackdependencies


function trackdependencies(model::Model{F}, args...) where {F}
    trace = trackmodel(model, args...)
    dependency_slice = strip_dependencies(strip_model_layers(F, trace))
    return makegraph(dependency_slice)
end


end # module
