function iterhash(itr)
    r = iterate(itr)
    isnothing(r) && throw(ArgumentError("`itr` cannot be empty!"))

    (el, state) = r
    return foldl((h, x) -> hash(x, h), Iterators.rest(itr, state); init = hash(el))
end
