@testset "not nested" begin
    let trie = VarTrie{Int}()

        # scalar
        trie[@varname(x)] = 1
        @test trie[@varname(x)] == 1
        
        trie[@varname(x)] *= 2
        @test trie[@varname(x)] == 2

        
        # getindex, initial & simple setindex!
        trie[@varname(y)] = [2, 3]
        @test trie[@varname(y)] == [2, 3]
        
        trie[@varname(y)] += [1, 1]
        @test trie[@varname(y)] == [3, 4]
        @test trie[@varname(y[1])] == 3
        @test trie[@varname(y[2])] == 4
        @test trie[@varname(y[:])] == [3, 4]
        @test trie[@varname(y[[2, 1]])] == [4, 3]

        trie[@varname(z)] = [1 3; 2 4]
        @test trie[@varname(z)] == [1 3; 2 4]
        
        trie[@varname(z)] *= 10
        @test trie[@varname(z)] == [10 30; 20 40]
        @test trie[@varname(z[1, 1])] == 10
        @test trie[@varname(z[2, 1])] == 20
        @test trie[@varname(z[1, 2])] == 30
        @test trie[@varname(z[2, 2])] == 40
        @test trie[@varname(z[4])] == 40
        @test trie[@varname(z[:])] == [10, 20, 30, 40]
        @test trie[@varname(z[[3, 2]])] == [30, 20]
        @test trie[@varname(z[:, 1])] == [10, 20]
        @test trie[@varname(z[2, :])] == [20, 40]

        
        # fancy setindex!
        trie[@varname(y[:])] = [2, 3]
        @test trie[@varname(y)] == [2, 3]
        
        trie[@varname(y[1])] = 100
        @test trie[@varname(y)] == [100, 3]

        trie[@varname(y[[2, 1]])] = [20, 10]
        @test trie[@varname(y)] == [10, 20]

        trie[@varname(z[:])] = [10 30; 20 40]
        @test trie[@varname(z)] == [10 30; 20 40]

        trie[@varname(z[1, 1])] = 100
        @test trie[@varname(z)] == [100 30; 20 40]
        
        trie[@varname(z[4])] = 400
        @test trie[@varname(z)] == [100 30; 20 400]
        
        trie[@varname(z[[3, 2]])] = [300, 200]
        @test trie[@varname(z)] == [100 300; 200 400]

        trie[@varname(z[:, 1])] /= 10 
        @test trie[@varname(z)] == [10 300; 20 400]

        trie[@varname(z[2, :])] /= 10 
        @test trie[@varname(z)] == [10 300; 2 40]
    end
end


@testset "nested" begin
    let trie = VarTrie{Int}()
        # start with an indexed variable
        # trie[@varname(x[:])] = [1, 2]

        # trie[@varname(y[1])] = 1
        # trie[@varname(y[2:5])] = 2:5
    end
end
