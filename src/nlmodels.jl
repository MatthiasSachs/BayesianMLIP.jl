
module NLModels

using ACE
using ACE: evaluate
using ACEatoms
using JuLIP
using NeighbourLists

export FSModel

struct FSModel
    basis1
    basis2
    rcut
    transform
    c1
    c2
end

function ACE.evaluate(m::FSModel, at)
    nlist = neighbourlist(at, m.rcut)
    part1,part2 = 0.0,0.0
    for k = 1:length(at)
        _, Rs = NeighbourLists.neigs(nlist, k)
        cfg = [ ACE.State(rr = r)  for r in Rs ] |> ACEConfig
        B1 = ACE.evaluate(m.basis1, cfg) 
        B2 = ACE.evaluate(m.basis2, cfg) 
        part1 += sum( c*b for (c,b) in zip(m.c1,B1)) 
        part2 += m.transform(sum( c*b for (c,b) in zip(m.c2,B2)).val)
    end
    return part1 + part2


end

end
#%%

