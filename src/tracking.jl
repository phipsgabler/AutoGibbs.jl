using IRTracker
using DynamicPPL

export trackmodel, strip_model_layers


struct AutoGibbsContext{F} <: AbstractTrackingContext end

IRTracker.canrecur(ctx::AutoGibbsContext, ::Model, args...) = true
IRTracker.canrecur(ctx::AutoGibbsContext{F}, f::F, args...) where {F} = true
IRTracker.canrecur(ctx::AutoGibbsContext, f, args...) = false

trackmodel(model::Model{F}) where {F} = track(AutoGibbsContext{F}(), model)


function strip_model_layers(node::NestedCallNode{<:Any, F, <:Tuple{<:Model{F}, Vararg}}) where F
    return node
end

function strip_model_layers(node::NestedCallNode{<:Any, <:Model{F}}) where {F}
    ix = findfirst(c -> c isa NestedCallNode{<:Any, <:Union{<:Model{F}, F}}, getchildren(node))
    return strip_model_layers(getchildren(node)[ix])
end
