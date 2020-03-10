using IRTracker
using DynamicPPL

import IRTracker: canrecur, trackedcall, trackednested


struct AutoGibbsContext <: AbstractTrackingContext
    inmodel::Bool
end

AutoGibbsContext() = AutoGibbsContext(true)

canrecur(ctx::AutoGibbsContext, f, args...) = ctx.inmodel
update_context(ctx::AutoGibbsContext, ::TapeExpr{<:Model}) = AutoGibbsContext(false)
update_context(ctx::AutoGibbsContext, ::TapeExpr) = ctx

function trackednested(ctx::AutoGibbsContext,
                       f_repr::TapeExpr,
                       args_repr::ArgumentTuple{TapeValue},
                       info::NodeInfo)
    recordnestedcall(update_context(ctx, f_repr), f_repr, args_repr, info)
end

# function trackedcall(::AutoGibbsContext,
#                      f_repr::TapeExpr{<:Model},
#                      args_repr::ArgumentTuple{TapeValue},
#                      info::NodeInfo)
    
# end

trackmodel(model::Model) =
    track(AutoGibbsContext(), model.f, VarInfo(), SampleFromPrior(), DefaultContext(), model)
