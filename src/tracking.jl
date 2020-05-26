using IRTracker
using DynamicPPL

export trackmodel, strip_model_layers


# F is always the type of the model evaluator closure

# dispatching on values
ismodelcall(::Type{F}, f, args...) where {F} = ismodelcall(F, typeof(f), typeof.(args)...)
ismodelcall(::Type{F}, ::NestedCallNode{<:Any, G, TArgs}) where {F, G, TArgs} = ismodelcall(F, G, TArgs.parameters...)

# dispatching on types
ismodelcall(::Type{F}, ::Type{<:Model{F}}, ::Vararg{<:Type}) where {F} = true
ismodelcall(::Type{F}, ::Type{typeof(Core._apply)}, ::Type{<:Model{F}}, ::Vararg{<:Type}) where {F} = true
ismodelcall(::Type{F}, ::Type{typeof(DynamicPPL.evaluate_threadsafe)}, ::Type, ::Type{<:Model{F}}, ::Vararg{<:Type}) where {F} = true
ismodelcall(::Type{F}, ::Type{typeof(DynamicPPL.evaluate_threadunsafe)}, ::Type, ::Type{<:Model{F}}, ::Vararg{<:Type}) where {F} = true
ismodelcall(::Type{F}, ::Type{F}, ::Vararg{<:Type}) where {F} = true
ismodelcall(::Type{F}, ::Type{G}, ::Vararg{<:Type}) where {F, G} = false


struct AutoGibbsContext{F} <: AbstractTrackingContext end

IRTracker.canrecur(ctx::AutoGibbsContext{F}, f, args...) where {F} = ismodelcall(F, f, args...)


trackmodel(model::Model{F}) where {F} = track(AutoGibbsContext{F}(), model)


function strip_model_layers(::Type{F}, node::NestedCallNode{<:Any, F}) where {F}
    # base case: call of the model closure, F
    return node
end

function strip_model_layers(::Type{F}, node::NestedCallNode) where {F}
    if ismodelcall(F, node)
        children = getchildren(node)
        ix = findfirst(Base.Fix1(ismodelcall, F), children)
        
        if ix !== nothing
            return strip_model_layers(F, children[ix])
        else
            throw(ArgumentError("$node does not contain any model call, you should not have reached this point!"))
        end
    else
        throw(ArgumentError("$node is not a model call, you should not have reached this point!"))
    end
end
