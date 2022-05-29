
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
    c1  # = vector a, with length K
    c2  # = vector a', with length K'
end

function ACE.evaluate(m::FSModel, at) 
    # neighbourlist computes all the relevant particles within the rcut radius for each particle. 
    nlist = neighbourlist(at, m.rcut) 
    print(typeof(nlist))        #Type: PairList{Float64, Int64} 
    print(length(nlist.X))      # 54-vector with each element a 3-vector
    print(fieldnames(typeof(nlist)))       # fieldnames (:X, :C, :cutoff, :i, :j, :S, :first)

    lin_part, nonlin_part = 0.0,0.0
    for i = 1:length(at)       # i index in the summation, indexing over number of particles 
        _, Rs = NeighbourLists.neigs(nlist, i)
        cfg = [ ACE.State(rr = r) for r in Rs ] |> ACEConfig

        # Inputs particle data r_{ij} & evaluates values of B_k ({r_{ij}}), B'_k ({r_{ij}}) for all K, K' basis functions
        B1 = ACE.evaluate(m.basis1, cfg) # For each i, B1 is a K-vector with elements B_1 ({r_{ij}}) ... B_K ({r_{ij}})
        B2 = ACE.evaluate(m.basis2, cfg) # For each i, B1 is a K-vector with elements B_1 ({r_{ij}}) ... B_K' ({r_{ij}})

        lin_part += sum( c*b for (c,b) in zip(m.c1,B1))                         # Sum up a_k B_k ({r_{ij}}) over index k=1 to K
        nonlin_part += m.transform(sum( c*b for (c,b) in zip(m.c2,B2)).val)     # Sum up a'_k B'_k ({r_{ij}}) over k=1 to K'& transform with F
    end
    return lin_part + nonlin_part
end

#Implement Gradient w.r.t. position vectors
function forces!(F, m::FSModel, at::Atoms; nlist = nothing) 
    if nlist === nothing
        nlist = neighbourlist(at, m.rcut)
    end
    for k = 1:length(at)
        _, Rs = NeighbourLists.neigs(nlist, k)
        cfg = [ ACE.State(rr = r)  for r in Rs ] |> ACEConfig
        F[k] += - sum(m.c1 .* ACE.evaluate_d(m.basis1, cfg)) # ACE.evaluate_d = (\nabla_{r_k} B_l)_{l=1}^{N_{basis}}
        F[k] += - sum(m.c2 .* ACE.evaluate_d(m.basis2, cfg)) * m.transform_d(sum(m.c2.*ACE.evaluate(m.basis2, cfg))) 
    end
end

function allocate_F(n::Int)
    return zeros(SVector{Float64,3}, n)
end

function forces(m::FSModel, at::Atoms; nlist = nothing) 
    # Compute gradient of the potential w.r.t. position vectors
    F = allocate_F(length(at))
    forces!(F, m::FSModel, at::Atoms; nlist=nlist) 
    return F
end


# Gradient of Finnis-Sinclair w.r.t. parameter c1 (a) and c2 (a') 
function evaluate_param_d(m::FSModel, at)
    nlist = neighbourlist(at, m.rcut)
    grad1 = zeros(length(m.basis1))     # gradient w.r.t. c1 
    grad2 = zeros(length(m.basis2))     # gradient w.r.t. c2 
    for i = 1:length(at)
        _, Rs = NeighbourLists.neigs(nlist, i)
        Ri = [ ACE.State(rr = r)  for r in Rs ] |> ACEConfig
        B1 = ACE.evaluate(m.basis1, Ri) # evaluate basis1 at atomic environement Ri
        B2 = ACE.evaluate(m.basis2, Ri) # evaluate basis2 at atomic environement Ri
        grad1 += [ b.val for b in B1]
        grad2 += [ b.val for b in B2] * m.transform_d(sum( c* b for (c,b) in zip(m.c2,B2)).val)
    end
    return grad1, grad2
end


end
#%%