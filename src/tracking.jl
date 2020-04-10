using IRTracker
using DynamicPPL

struct AutoGibbsContext{F} <: AbstractTrackingContext end

IRTracker.canrecur(ctx::AutoGibbsContext, ::Model, args...) = true
IRTracker.canrecur(ctx::AutoGibbsContext{F}, f::F, args...) where {F} = true
IRTracker.canrecur(ctx::AutoGibbsContext, f, args...) = false

trackmodel(model::Model{F}) where {F} = track(AutoGibbsContext{F}(), model)

function strip_calls(node::NestedCallNode)
    
end

export trackmodel
