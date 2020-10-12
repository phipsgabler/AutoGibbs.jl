# MacroTools.prewalk(MacroTools.rmlines, MacroTools.gensym_ids(expr2)) |> MacroTools.flatten


coinflip_1(; y) = coinflip_1(y)

function coinflip_1(y)
    function inner_function_2(
        vi_3::DynamicPPL.VarInfo,
        sampler_4::DynamicPPL.AbstractSampler,
        ctx_5::DynamicPPL.AbstractContext,
        model_6
    )
        
        local y
        temp_var_7 = model_6.args.y
        varT_8 = typeof(temp_var_7)
        
        if temp_var_7 isa DynamicPPL.FloatOrArrayType
            y = DynamicPPL.get_matching_type(sampler_4, vi_3, temp_var_7)
        elseif DynamicPPL.hasmissing(varT_8)
            y = (DynamicPPL.get_matching_type(sampler_4, vi_3, varT_8))(temp_var_7)
        else
            y = temp_var_7
        end
        
        vi_3.logp = 0
        temp_right_9 = Beta(1, 1)
        DynamicPPL.assert_dist(temp_right_9, msg="Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions on line 449.")
        
        preprocessed_10 = begin
            sym_11 = Val(:p)
            if !(DynamicPPL.inparams(sym_11, Val{(:y,)}())) || DynamicPPL.inparams(sym_11, DynamicPPL.getmissing(model_6))
                (DynamicPPL.VarName{:p}(""), ())
            else
                if DynamicPPL.inparams(sym_11, Val{(:y,)}())
                    lhs_12 = p
                    if lhs_12 === missing
                        (DynamicPPL.VarName{:p}(""), ())
                    else
                        lhs_12
                    end
                else
                    throw("This point should not be reached. Please report this error.")
                end
            end
        end
        
        if preprocessed_10 isa Tuple
            (vn_13, inds_14) = preprocessed_10
            out_15 = DynamicPPL.tilde(ctx_5, sampler_4, temp_right_9, vn_13, inds_14, vi_3)
            p = out_15[1]
            vi_3.logp += out_15[2]
        else
            vi_3.logp += DynamicPPL.tilde(ctx_5, sampler_4, temp_right_9, preprocessed_10, vi_3)
        end
        
        N = length(y)
        
        for n = 1:N
            temp_right_16 = Bernoulli(p)
            DynamicPPL.assert_dist(temp_right_16, msg="Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions on line 449.")
            preprocessed_17 = begin
                sym_18 = Val(:y)
                if !(DynamicPPL.inparams(sym_18, Val{(:y,)}())) || DynamicPPL.inparams(sym_18, DynamicPPL.getmissing(model_6))
                    (DynamicPPL.VarName{:y}(foldl(*, ("[" * join([if n === (:)
                                                                  "Colon()"
                                                                  else
                                                                  string(n)
                                                                  end], ",") * "]",), init="")), ((n,),))
                else
                    if DynamicPPL.inparams(sym_18, Val{(:y,)}())
                        lhs_19 = y[n]
                        if lhs_19 === missing
                            (DynamicPPL.VarName{:y}(foldl(*, ("[" * join([if n === (:)
                                                                          "Colon()"
                                                                          else
                                                                          string(n)
                                                                          end], ",") * "]",), init="")),
                             ((n,),))
                        else
                            lhs_19
                        end
                    else
                        throw("This point should not be reached. Please report this error.")
                    end
                end
            end
            
            if preprocessed_17 isa Tuple
                (vn_20, inds_21) = preprocessed_17
                out_22 = DynamicPPL.tilde(ctx_5, sampler_4, temp_right_16, vn_20, inds_21, vi_3)
                y[n] = out_22[1]
                vi_3.logp += out_22[2]
            else
                vi_3.logp += DynamicPPL.tilde(ctx_5, sampler_4, temp_right_16, preprocessed_17, vi_3)
            end
        end
    end
    
    return DynamicPPL.Model(inner_function_2,
                            DynamicPPL.namedtuple(NamedTuple{(:y,), Tuple{DynamicPPL.get_type(y)}}, (y,)),
                            DynamicPPL.ModelGen{(:y,)}(coinflip_1, NamedTuple()))
end

coinflip = DynamicPPL.ModelGen{(:y,)}(coinflip_1, NamedTuple())
