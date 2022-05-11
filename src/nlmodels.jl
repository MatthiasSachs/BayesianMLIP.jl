
module NLModels

using ACE
using ACE: evaluate
using ACEatoms
using JuLIP
using NeighbourLists

export FSModel

struct FSModel
    basis1 # = B 
    basis2 # = B'
    rcut # cutoff radius of the ace bases B, B'
    transform   # = - Sqrt 
    transform_d # = 1 / (2 Sqrt)  
    c1  # = a
    c2  # = a'
end

function ACE.evaluate(m::FSModel, at)
    nlist = neighbourlist(at, m.rcut)
    lin_part, nonlin_part = 0.0,0.0
    for k = 1:length(at)
        _, Rs = NeighbourLists.neigs(nlist, k)
        cfg = [ ACE.State(rr = r)  for r in Rs ] |> ACEConfig
        B1 = ACE.evaluate(m.basis1, cfg) 
        B2 = ACE.evaluate(m.basis2, cfg) 
        lin_part += sum( c*b for (c,b) in zip(m.c1,B1)) 
        nonlin_part += m.transform(sum( c*b for (c,b) in zip(m.c2,B2)).val)
    end
    return lin_part + nonlin_part
end


function ACE.evaluate_d(m::FSModel, at)
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

