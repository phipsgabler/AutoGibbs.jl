# using Turing
using Distributions
using DynamicPPL
using IRTracker

@model coinflip(y) = begin
    p ~ Beta(1, 1)
    N = length(y)
    for i = 1:N
        y[i] ~ Bernoulli(p)
    end
end

model = coinflip([1,1,0])
vi = VarInfo()
# model(vi)



# ⟨model⟩(⟨vi⟩, ()...) = nothing
#   @1: [Arg:§1:%1] model
#   @2: [Arg:§1:%2] vi
#   @3: [§1:%3] ⟨SampleFromPrior⟩(, ()...) = SampleFromPrior()
#     @1: [Arg:§1:%1] @3#1 = SampleFromPrior
#     @2: [Const:§1:%2] 
#     @3: [§1:%3] new(@2) = SampleFromPrior()
#     @4: [§1:&1] return @3 = SampleFromPrior()
#   @4: [§1:%4] @1(@2, @3, ()...) = nothing
#     @1: [Arg:§1:%1] @4#1 = model
#     @2: [Arg:§1:%2] @4#2 = vi
#     @3: [Arg:§1:%3] @4#3 = SampleFromPrior()
#     @4: [§1:%4] ⟨DefaultContext⟩(, ()...) = DefaultContext()
#       @1: [Arg:§1:%1] @4#1 = DefaultContext
#       @2: [Const:§1:%2] 
#       @3: [§1:%3] new(@2) = DefaultContext()
#       @4: [§1:&1] return @3 = DefaultContext()
#     @5: [§1:%5] @1(, (@2, @3, @4)...) = nothing
#       @1: [Arg:§1:%1] @5#1 = model
#       @2: [Arg:§1:%2] @5#2 = (vi, SampleFromPrior(), DefaultContext())
#       @3: [§1:%3] ⟨NamedTuple⟩(, ()...) = NamedTuple()
#       @4: [§1:%4] ⟨pairs⟩(@3, ()...) = Base.Iterators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}}()
#       @5: [§1:%5] ⟨tuple⟩(@4, @1) = (Base.Iterators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}}(), model)
#       @6: [§1:%6] ⟨Core._apply⟩(⟨DynamicPPL.#_#3⟩, @5, @2) = nothing
#       @7: [§1:&1] return @6 = nothing
#     @6: [§1:&1] return @5 = nothing
#   @5: [§1:&1] return @4 = nothing

trace = track(DepthLimitContext(2), model.f, vi, SampleFromPrior(), DefaultContext(), model);

# ⟨var"##inner_function#426#6"()⟩(⟨vi⟩, ⟨SampleFromPrior()⟩, ⟨DefaultContext()⟩, model) = nothing
#   @1: [Arg:§1:%1] var"##inner_function#426#6"()
#   @2: [Arg:§1:%2] vi
#   @3: [Arg:§1:%3] SampleFromPrior()
#   @4: [Arg:§1:%4] DefaultContext()
#   @5: [Arg:§1:%5] model
#   @6: [§1:%6] ⟨getproperty⟩(@5, ⟨:args⟩) = (y = [1, 1, 0],)
#   @7: [§1:%7] ⟨getproperty⟩(@6, ⟨:y⟩) = [1, 1, 0]
#   @8: [§1:%8] ⟨typeof⟩(@7) = Array{Int64,1}
#   @9: [Const:§1:%9] Type{#s23} where #s23<:Union{AbstractFloat, AbstractArray}
#   @10: [§1:%10] ⟨isa⟩(@7, @9) = false
#   @11: [§1:&1] goto §3 since @10 == false
#   @12: [Const:§3:%13] DynamicPPL.hasmissing
#   @13: [§3:%14] @12(@8) = false
#   @14: [§3:&1] goto §5 since @13 == false
#   @15: [§5:&1] goto §6 (@7)
#   @16: [Arg:§6:%18] @15#1 = [1, 1, 0]
#   @17: [Const:§6:%19] DynamicPPL.resetlogp!
#   @18: [§6:%20] @17(@2) = 0
#   @19: [§6:%21] ⟨Beta⟩(⟨1⟩, ⟨1⟩) = Beta{Float64}(α=1.0, β=1.0)
#   @20: [Const:§6:%22] (:msg,)
#   @21: [§6:%23] ⟨Core.apply_type⟩(⟨NamedTuple⟩, @20) = NamedTuple{(:msg,),T} where T<:Tuple
#   @22: [§6:%24] ⟨tuple⟩(⟨"Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions on line 340."⟩) = ("Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions on line 340.",)
#   @23: [§6:%25] @21(@22) = (msg = "Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions on line 340.",)
#   @24: [Const:§6:%26] DynamicPPL.assert_dist
#   @25: [§6:%27] ⟨Core.kwfunc⟩(@24) = DynamicPPL.var"#kw##assert_dist"()
#   @26: [Const:§6:%28] DynamicPPL.assert_dist
#   @27: [§6:%29] @25(@23, @26, @19) = true
#   @28: [§6:%30] ⟨Val⟩(⟨:p⟩) = Val{:p}()
#   @29: [Const:§6:%31] DynamicPPL.inparams
#   @30: [§6:%32] @29(@28, ⟨Val{(:y,)}()⟩) = false
#   @31: [§6:%33] ⟨!⟩(@30) = true
#   @32: [§6:&2] goto §7
#   @33: [§7:&1] goto §9 (@31)
#   @34: [Arg:§9:%38] @33#1 = true
#   @35: [§9:&2] goto §10
#   @36: [Const:§10:%39] VarName
#   @37: [§10:%40] ⟨Core.apply_type⟩(@36, ⟨:p⟩) = VarName{:p}
#   @38: [§10:%41] @37(⟨""⟩) = VarName{:p}("")
#   @39: [§10:%42] ⟨tuple⟩(@38, ⟨()⟩) = (VarName{:p}(""), ())
#   @40: [§10:&1] goto §18 (@39)
#   @41: [Arg:§18:%53] @40#1 = (VarName{:p}(""), ())
#   @42: [§18:%54] ⟨isa⟩(@41, ⟨Tuple⟩) = true
#   @43: [§18:&2] goto §19
#   @44: [§19:%55] ⟨Base.indexed_iterate⟩(@41, ⟨1⟩) = (VarName{:p}(""), 2)
#   @45: [§19:%56] ⟨getfield⟩(@44, ⟨1⟩) = VarName{:p}("")
#   @46: [§19:%57] ⟨getfield⟩(@44, ⟨2⟩) = 2
#   @47: [§19:%58] ⟨Base.indexed_iterate⟩(@41, ⟨2⟩, @46) = ((), 3)
#   @48: [§19:%59] ⟨getfield⟩(@47, ⟨1⟩) = ()
#   @49: [Const:§19:%60] DynamicPPL.tilde
#   @50: [§19:%61] @49(@4, @3, @19, @45, @48, @2) = (0.8047069487761294, -0.0)
#   @51: [§19:%62] ⟨getindex⟩(@50, ⟨1⟩) = 0.8047069487761294
#   @52: [Const:§19:%63] DynamicPPL.acclogp!
#   @53: [§19:%64] ⟨getindex⟩(@50, ⟨2⟩) = -0.0
#   @54: [§19:%65] @52(@2, @53) = 0.0
#   @55: [§19:&1] goto §21 (@51)
#   @56: [Arg:§21:%70] @55#1 = 0.8047069487761294
#   @57: [§21:%71] ⟨length⟩(@16) = 3
#   @58: [§21:%72] ⟨Colon()⟩(⟨1⟩, @57) = 1:3
#   @59: [§21:%73] ⟨iterate⟩(@58) = (1, 1)
#   @60: [§21:%74] ⟨===⟩(@59, ⟨nothing⟩) = false
#   @61: [§21:%75] ⟨not_int⟩(@60) = true
#   @62: [§21:&2] goto §22 (@59)
#   @63: [Arg:§22:%76] @62#1 = (1, 1)
#   @64: [§22:%77] ⟨getfield⟩(@63, ⟨1⟩) = 1
#   @65: [§22:%78] ⟨getfield⟩(@63, ⟨2⟩) = 1
#   @66: [§22:%79] ⟨Bernoulli⟩(@56) = Bernoulli{Float64}(p=0.8047069487761294)
#   @67: [Const:§22:%80] (:msg,)
#   @68: [§22:%81] ⟨Core.apply_type⟩(⟨NamedTuple⟩, @67) = NamedTuple{(:msg,),T} where T<:Tuple
#   @69: [§22:%82] ⟨tuple⟩(⟨"Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions on line 340."⟩) = ("Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions on line 340.",)
#   @70: [§22:%83] @68(@69) = (msg = "Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions on line 340.",)
#   @71: [Const:§22:%84] DynamicPPL.assert_dist
#   @72: [§22:%85] ⟨Core.kwfunc⟩(@71) = DynamicPPL.var"#kw##assert_dist"()
#   @73: [Const:§22:%86] DynamicPPL.assert_dist
#   @74: [§22:%87] @72(@70, @73, @66) = true
#   @75: [§22:%88] ⟨Val⟩(⟨:y⟩) = Val{:y}()
#   @76: [Const:§22:%89] DynamicPPL.inparams
#   @77: [§22:%90] @76(@75, ⟨Val{(:y,)}()⟩) = true
#   @78: [§22:%91] ⟨!⟩(@77) = false
#   @79: [§22:&1] goto §24 since @78 == false
#   @80: [Const:§24:%92] DynamicPPL.inparams
#   @81: [Const:§24:%93] DynamicPPL.getmissing
#   @82: [§24:%94] @81(@5) = Val{()}()
#   @83: [§24:%95] @80(@75, @82) = false
#   @84: [§24:&1] goto §25 (@83)
#   @85: [Arg:§25:%96] @84#1 = false
#   @86: [§25:&1] goto §30 since @85 == false
#   @87: [Const:§30:%116] DynamicPPL.inparams
#   @88: [§30:%117] @87(@75, ⟨Val{(:y,)}()⟩) = true
#   @89: [§30:&2] goto §31
#   @90: [§31:%118] ⟨getindex⟩(@16, @64) = 1
#   @91: [§31:%119] ⟨===⟩(@90, ⟨missing⟩) = false
#   @92: [§31:&1] goto §36 since @91 == false
#   @93: [§36:&1] goto §37 (@90)
#   @94: [Arg:§37:%139] @93#1 = 1
#   @95: [§37:&1] goto §39 (@94)
#   @96: [Arg:§39:%141] @95#1 = 1
#   @97: [§39:&1] goto §40 (@96)
#   @98: [Arg:§40:%142] @97#1 = 1
#   @99: [§40:%143] ⟨isa⟩(@98, ⟨Tuple⟩) = false
#   @100: [§40:&1] goto §42 since @99 == false
#   @101: [Const:§42:%156] DynamicPPL.acclogp!
#   @102: [Const:§42:%157] DynamicPPL.tilde
#   @103: [§42:%158] @102(@4, @3, @66, @98, @2) = -0.21727710662919572
#   @104: [§42:%159] @101(@2, @103) = -0.21727710662919572
#   @105: [§42:&1] goto §43
#   @106: [§43:%160] ⟨iterate⟩(@58, @65) = (2, 2)
#   @107: [§43:%161] ⟨===⟩(@106, ⟨nothing⟩) = false
#   @108: [§43:%162] ⟨not_int⟩(@107) = true
#   @109: [§43:&2] goto §22 (@106)
#   @110: [Arg:§22:%76] @109#1 = (2, 2)
#   @111: [§22:%77] ⟨getfield⟩(@110, ⟨1⟩) = 2
#   @112: [§22:%78] ⟨getfield⟩(@110, ⟨2⟩) = 2
#   @113: [§22:%79] ⟨Bernoulli⟩(@56) = Bernoulli{Float64}(p=0.8047069487761294)
#   @114: [Const:§22:%80] (:msg,)
#   @115: [§22:%81] ⟨Core.apply_type⟩(⟨NamedTuple⟩, @114) = NamedTuple{(:msg,),T} where T<:Tuple
#   @116: [§22:%82] ⟨tuple⟩(⟨"Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions on line 340."⟩) = ("Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions on line 340.",)
#   @117: [§22:%83] @115(@116) = (msg = "Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions on line 340.",)
#   @118: [Const:§22:%84] DynamicPPL.assert_dist
#   @119: [§22:%85] ⟨Core.kwfunc⟩(@118) = DynamicPPL.var"#kw##assert_dist"()
#   @120: [Const:§22:%86] DynamicPPL.assert_dist
#   @121: [§22:%87] @119(@117, @120, @113) = true
#   @122: [§22:%88] ⟨Val⟩(⟨:y⟩) = Val{:y}()
#   @123: [Const:§22:%89] DynamicPPL.inparams
#   @124: [§22:%90] @123(@122, ⟨Val{(:y,)}()⟩) = true
#   @125: [§22:%91] ⟨!⟩(@124) = false
#   @126: [§22:&1] goto §24 since @125 == false
#   @127: [Const:§24:%92] DynamicPPL.inparams
#   @128: [Const:§24:%93] DynamicPPL.getmissing
#   @129: [§24:%94] @128(@5) = Val{()}()
#   @130: [§24:%95] @127(@122, @129) = false
#   @131: [§24:&1] goto §25 (@130)
#   @132: [Arg:§25:%96] @131#1 = false
#   @133: [§25:&1] goto §30 since @132 == false
#   @134: [Const:§30:%116] DynamicPPL.inparams
#   @135: [§30:%117] @134(@122, ⟨Val{(:y,)}()⟩) = true
#   @136: [§30:&2] goto §31
#   @137: [§31:%118] ⟨getindex⟩(@16, @111) = 1
#   @138: [§31:%119] ⟨===⟩(@137, ⟨missing⟩) = false
#   @139: [§31:&1] goto §36 since @138 == false
#   @140: [§36:&1] goto §37 (@137)
#   @141: [Arg:§37:%139] @140#1 = 1
#   @142: [§37:&1] goto §39 (@141)
#   @143: [Arg:§39:%141] @142#1 = 1
#   @144: [§39:&1] goto §40 (@143)
#   @145: [Arg:§40:%142] @144#1 = 1
#   @146: [§40:%143] ⟨isa⟩(@145, ⟨Tuple⟩) = false
#   @147: [§40:&1] goto §42 since @146 == false
#   @148: [Const:§42:%156] DynamicPPL.acclogp!
#   @149: [Const:§42:%157] DynamicPPL.tilde
#   @150: [§42:%158] @149(@4, @3, @113, @145, @2) = -0.21727710662919572
#   @151: [§42:%159] @148(@2, @150) = -0.43455421325839144
#   @152: [§42:&1] goto §43
#   @153: [§43:%160] ⟨iterate⟩(@58, @112) = (3, 3)
#   @154: [§43:%161] ⟨===⟩(@153, ⟨nothing⟩) = false
#   @155: [§43:%162] ⟨not_int⟩(@154) = true
#   @156: [§43:&2] goto §22 (@153)
#   @157: [Arg:§22:%76] @156#1 = (3, 3)
#   @158: [§22:%77] ⟨getfield⟩(@157, ⟨1⟩) = 3
#   @159: [§22:%78] ⟨getfield⟩(@157, ⟨2⟩) = 3
#   @160: [§22:%79] ⟨Bernoulli⟩(@56) = Bernoulli{Float64}(p=0.8047069487761294)
#   @161: [Const:§22:%80] (:msg,)
#   @162: [§22:%81] ⟨Core.apply_type⟩(⟨NamedTuple⟩, @161) = NamedTuple{(:msg,),T} where T<:Tuple
#   @163: [§22:%82] ⟨tuple⟩(⟨"Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions on line 340."⟩) = ("Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions on line 340.",)
#   @164: [§22:%83] @162(@163) = (msg = "Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions on line 340.",)
#   @165: [Const:§22:%84] DynamicPPL.assert_dist
#   @166: [§22:%85] ⟨Core.kwfunc⟩(@165) = DynamicPPL.var"#kw##assert_dist"()
#   @167: [Const:§22:%86] DynamicPPL.assert_dist
#   @168: [§22:%87] @166(@164, @167, @160) = true
#   @169: [§22:%88] ⟨Val⟩(⟨:y⟩) = Val{:y}()
#   @170: [Const:§22:%89] DynamicPPL.inparams
#   @171: [§22:%90] @170(@169, ⟨Val{(:y,)}()⟩) = true
#   @172: [§22:%91] ⟨!⟩(@171) = false
#   @173: [§22:&1] goto §24 since @172 == false
#   @174: [Const:§24:%92] DynamicPPL.inparams
#   @175: [Const:§24:%93] DynamicPPL.getmissing
#   @176: [§24:%94] @175(@5) = Val{()}()
#   @177: [§24:%95] @174(@169, @176) = false
#   @178: [§24:&1] goto §25 (@177)
#   @179: [Arg:§25:%96] @178#1 = false
#   @180: [§25:&1] goto §30 since @179 == false
#   @181: [Const:§30:%116] DynamicPPL.inparams
#   @182: [§30:%117] @181(@169, ⟨Val{(:y,)}()⟩) = true
#   @183: [§30:&2] goto §31
#   @184: [§31:%118] ⟨getindex⟩(@16, @158) = 0
#   @185: [§31:%119] ⟨===⟩(@184, ⟨missing⟩) = false
#   @186: [§31:&1] goto §36 since @185 == false
#   @187: [§36:&1] goto §37 (@184)
#   @188: [Arg:§37:%139] @187#1 = 0
#   @189: [§37:&1] goto §39 (@188)
#   @190: [Arg:§39:%141] @189#1 = 0
#   @191: [§39:&1] goto §40 (@190)
#   @192: [Arg:§40:%142] @191#1 = 0
#   @193: [§40:%143] ⟨isa⟩(@192, ⟨Tuple⟩) = false
#   @194: [§40:&1] goto §42 since @193 == false
#   @195: [Const:§42:%156] DynamicPPL.acclogp!
#   @196: [Const:§42:%157] DynamicPPL.tilde
#   @197: [§42:%158] @196(@4, @3, @160, @192, @2) = -1.6332540217433904
#   @198: [§42:%159] @195(@2, @197) = -2.067808235001782
#   @199: [§42:&1] goto §43
#   @200: [§43:%160] ⟨iterate⟩(@58, @159) = nothing
#   @201: [§43:%161] ⟨===⟩(@200, ⟨nothing⟩) = true
#   @202: [§43:%162] ⟨not_int⟩(@201) = false
#   @203: [§43:&1] goto §44 since @202 == false
#   @204: [§44:&1] return ⟨nothing⟩
