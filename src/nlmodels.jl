module NLModels

using ACE 
using ACE: evaluate, val
using ACEatoms
using JuLIP
import JuLIP: forces, energy
using NeighbourLists
using Random: seed!
using LinearAlgebra: dot
using Zygote
using StaticArrays

#using ACE, ACEbase, Test, ACE.Testing
#using ACE: evaluate, SymmetricBasis, PIBasis, O3, State, val 
#import ACE: nparams, set_params!, params,val 
export FSModel, energy, forces,  hamiltonian, kenergy

struct FSModel{T}
    model1::ACE.LinearACEModel
    model2::ACE.LinearACEModel
    transform
    rcut::T
end

FSModel(model1::ACE.LinearACEModel, model2::ACE.LinearACEModel, rcut::T) where {T<:Real} = FSModel(model1::ACE.LinearACEModel, model2::ACE.LinearACEModel, props -> sum( (1 .+ val.(props).^2).^0.5 ), rcut::T) 



function JuLIP.energy(m::FSModel, at::Atoms) 
    FS2 = m.transform
    nlist = neighbourlist(at, m.rcut)
    E =  ACE.Invariant(0.0)
    for k = 1:length(at)
        Js, Rs = NeighbourLists.neigs(nlist, k)    # Js = indices, Rs = PositionVectors 
        cfg = ACEConfig( [ ACE.State(rr = r)  for r in Rs ] )
        E += FS2(evaluate(m.model2, cfg))
        E += evaluate(m.model1, cfg)[1]
    end
    return E
end

function JuLIP.forces(m::FSModel, at::Atoms) 
    nlist = neighbourlist(at, m.rcut)
    F = zeros(SVector{3,Float64}, length(at))
    for k = 1:length(at)
        Js, Rs = NeighbourLists.neigs(nlist, k)    # Js = indices, Rs = PositionVectors 
        cfg = ACEConfig( [ ACE.State(rr = r)  for r in Rs ] )
        g1 = Zygote.gradient(x ->  m.transform(evaluate(m.model2, x)), cfg)[1]
        g2 = ACE.grad_config(m.model1, cfg)
        for (i,j) in enumerate(Js)
            F[j] += -g1[i].rr - g2[i][1].rr
            F[k] +=  g1[i].rr + g2[i][1].rr
        end
    end
    return F
end


function allocate_F(n::Int)
    return zeros(SVector{3, Float64}, n)
end


# mutable struct FSModel      # <: Any
#     basis1 # = B 
#     basis2 # = B'
#     rcut # cutoff radius of the ace bases B, B'
#     transform   # = - Sqrt 
#     transform_d # = 1 / (2 Sqrt)  
#     c1  # = vector a, with length K
#     c2  # = vector a', with length K'
# end

# function energy(m::FSModel, at::Atoms; nlist = nothing)
#     # Function representing our approximation model \hat{E} of our true Finnis-Sinclair potential E
#     # neighbourlist computes all the relevant particles within the rcut radius for each particle. 
#     nlist = neighbourlist(at, m.rcut)       # fieldnames (:X, :C, :cutoff, :i, :j, :S, :first)
#     println("Energy is 0")
#     lin_part, nonlin_part = 0.0,0.0
#     for i = 1:length(at)       # i index in the summation, indexing over number of particles 
#         _, Rs = NeighbourLists.neigs(nlist, i)
#         cfg = ACEConfig( [ ACE.State(rr = r) for r in Rs ] )

#         # Inputs particle data r_{ij} & evaluates values of B_k ({r_{ij}}), B'_k ({r_{ij}}) for all K, K' basis functions
#         B1 = ACE.evaluate(m.basis1, cfg) # For each i, B1 is a K-vector with elements B_1 ({r_{ij}}) ... B_K ({r_{ij}})
#         B2 = ACE.evaluate(m.basis2, cfg) # For each i, B1 is a K-vector with elements B_1 ({r_{ij}}) ... B_K' ({r_{ij}})

#         lin_part += sum( c*b for (c,b) in zip(m.c1,B1))                         # Sum up a_k B_k ({r_{ij}}) over index k=1 to K
#         nonlin_part += m.transform(sum( c*b for (c,b) in zip(m.c2,B2)).val)     # Sum up a'_k B'_k ({r_{ij}}) over k=1 to K'& transform with F
#     end
    
#     return (lin_part + nonlin_part).val
# end

# #Implement Gradient w.r.t. position vectors
# function forces!(F, m::FSModel, at::Atoms; nlist = nothing) 
#     if nlist === nothing
#         nlist = neighbourlist(at, m.rcut)
#     end
#     for k = 1:length(at)
#         _, Rs = NeighbourLists.neigs(nlist, k)
#         cfg = ACEConfig( [ ACE.State(rr = r)  for r in Rs ] ) 
#         F += - sum(m.c1 .* ACE.evaluate_d(m.basis1, cfg)) 
#         F += - sum(m.c2 .* ACE.evaluate_d(m.basis2, cfg)) * m.transform_d(sum(m.c2.*ACE.evaluate(m.basis2, cfg)).val) 
#     end
#     return F 
# end



# function allocate_F(n::Int)
#     println("Force is 0")
#     return zeros(ACE.SVector{3, Float64}, n)
# end

# function FS_paramGrad(m::FSModel, at)
#     # Compute gradient of Finnis-Sinclair potential w.r.t. parameters  c1 (a) and c2 (a') 

#     nlist = neighbourlist(at, m.rcut)
#     grad1 = zeros(length(m.basis1))     # gradient w.r.t. c1 
#     grad2 = zeros(length(m.basis2))     # gradient w.r.t. c2 
#     for i = 1:length(at)
#         _, Rs = NeighbourLists.neigs(nlist, i)
#         Ri = [ ACE.State(rr = r)  for r in Rs ] |> ACEConfig
#         B1 = ACE.evaluate(m.basis1, Ri) # evaluate basis1 at atomic environement Ri
#         B2 = ACE.evaluate(m.basis2, Ri) # evaluate basis2 at atomic environement Ri
#         grad1 += [ b.val for b in B1]
#         grad2 += [ b.val for b in B2] * m.transform_d(sum( c* b for (c,b) in zip(m.c2,B2)).val)
#     end
#     return grad1, grad2
# end


struct CombModel <: AbstractCalculator
    model_list
end

function forces(model::CombModel, at::Atoms)
    return sum(forces(m,at) for m in model.model_list)
end

function energy(model::CombModel, at::Atoms)
    return sum(energy(m,at) for m in model.model_list)
end

function kenergy(at::Atoms)
    return 0.5 * sum([dot(at.P[t] /at.M[t], at.P[t]) for t in 1:length(at.P)])
end

function hamiltonian(V, at::Atoms) 
    # Wish we would directly call this on outp, but this would require outp to store 
    # entire at object rather than at.X and at.P
    return energy(V, at) + kenergy(at::Atoms)
end 

end
