using Test

using AutoGibbs
using Distributions
using DynamicPPL
using Turing, Turing.RandomMeasures
using Random


include("utils.jl")


@testset "AutoGibbs.jl" begin
    @testset "dependencies" begin
        # include("test_dependencies.jl")
    end

    @testset "conditionals" begin
        include("test_conditionals.jl")
        
        @testset "IMM variants" begin
            include("test_imms.jl")
        end
    end
end
