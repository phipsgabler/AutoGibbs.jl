module AutoGibbs

include("tracking.jl")
include("dependencies.jl")
include("plotting.jl")
include("conditionals.jl")
include("auto_conditional.jl")


export trackdependencies


function slicedependencies(model::Model{F}, args...) where {F}
    trace = trackmodel(model, args...)
    slice = strip_dependencies(strip_model_layers(F, trace))
    return slice
end

function trackdependencies(model, args...)
    slice = slicedependencies(model, args...)
    return makegraph(slice)
end


end # module
