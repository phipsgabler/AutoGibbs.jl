using Test

using AutoGibbs
using Distributions
using DynamicPPL
using Turing, Turing.RandomMeasures


Turing.turnprogress(false)

include("utils.jl")


@testset "AutoGibbs.jl" begin
    @testset "dependencies" begin
        include("test_dependencies.jl")
    end

    @testset "conditionals" begin
        include("test_conditionals.jl")
    end

    @testset "vartrie" begin
        include("test_vartrie.jl")
    end
end
