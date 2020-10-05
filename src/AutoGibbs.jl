module AutoGibbs

include("tracking.jl")
include("dependencies.jl")
include("plotting.jl")
include("conditionals.jl")
include("auto_conditional.jl")
include("static_conditional.jl")
include("crp.jl")

export trackdependencies


function slicedependencies(model::Model{F}, args...) where {F}
    trace = trackmodel(model, args...)
    strip = strip_model_layers(F, trace)
    slice = strip_dependencies(strip)
    return slice
end

function trackdependencies(model, args...)
    slice = slicedependencies(model, args...)
    return makegraph(slice)
end


end # module
