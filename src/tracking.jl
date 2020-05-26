using IRTracker
using DynamicPPL

export trackmodel, strip_model_layers


const ModelEval = Union{typeof(DynamicPPL.evaluate_threadsafe),
                        typeof(DynamicPPL.evaluate_threadunsafe)}


struct AutoGibbsContext{F} <: AbstractTrackingContext end

IRTracker.canrecur(ctx::AutoGibbsContext{F}, ::Model{F}, args...) where {F} = true
IRTracker.canrecur(ctx::AutoGibbsContext{F}, ::typeof(Core._apply), ::Model{F}, args...) where {F} = true
IRTracker.canrecur(ctx::AutoGibbsContext{F}, ::ModelEval, rng, ::Model{F}, args...) where {F} = true
IRTracker.canrecur(ctx::AutoGibbsContext{F}, f::F, args...) where {F} = true
IRTracker.canrecur(ctx::AutoGibbsContext, f, args...) = false


trackmodel(model::Model{F}) where {F} = track(AutoGibbsContext{F}(), model)


ismodeleval(::NestedCallNode{<:Any, F, <:Tuple{<:Model{F}, Vararg}}) where {F} = true
ismodeleval(::NestedCallNode{<:Any, <:ModelEval, <:Tuple{<:Model, Vararg}}) = true
ismodeleval(::NestedCallNode{<:Any, <:Model}) = true
ismodeleval(::AbstractNode) = false

function strip_model_layers(node::NestedCallNode{<:Any, F, <:Tuple{<:Model{F}, Vararg}}) where {F}
    # base case: call of the model closure, F
    return node
end

function strip_model_layers(node::NestedCallNode)
    if ismodeleval(node)
        children = getchildren(node)
        ix = findfirst(ismodeleval, children)
        return strip_model_layers(children[ix])
    else
        throw(ArgumentError("$node is not a model call, you should not have reached this point!"))
    end
end
