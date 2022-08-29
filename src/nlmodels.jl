module NLModels
using ACE, ACEatoms, Flux, JuLIP, ACEflux
import ACEflux: FluxPotential
using ACE: evaluate, val, AbstractConfiguration
import JuLIP: forces, energy
using NeighbourLists
using Random: seed!
using LinearAlgebra: dot
using Zygote
using StaticArrays

import ACE: set_params!, nparams, params, evaluate, LinearACEModel, AbstractACEModel
export get_params, nparams, set_params!
export energy, forces, Hamiltonian, params, gradParams

# function get_params(pot::FluxPotential) 
#     return pot.model[1].weight
# end 

# function gradParams(pot, at::AbstractAtoms, θ)       # gradient of potential w.r.t. parameters
#     s = size(pot.model[1].weight)
#     pot.model[1].weight = reshape(θ, s[1], s[2])
#     p = Flux.params(pot.model)  
#     dE = Zygote.gradient(()->Energy(pot, at), p)
#     return dE[p[1]]
# end

function Hamiltonian(pot, at::AbstractAtoms) 
    KE = 0.5 * sum([dot(at.P[t] /at.M[t], at.P[t]) for t in 1:length(at.P)])
    return energy(pot, at) + KE
end 


mat2svecs(M::AbstractArray{T}, nc::Int) where {T} =   
      collect(reinterpret(SVector{nc, T}, M))
svecs2vec(M::AbstractVector{<: SVector{N, T}}) where {N, T} = 
      collect(reinterpret(T, M))

struct NLModel <: AbstractACEModel 
      m::LinearACEModel
      nc::Int
      σ
end

(nlm::NLModel)(cfg) = evaluate(nlm::NLModel, cfg)
ACE.evaluate(nlm::NLModel, cfg) = nlm.σ(evaluate(nlm.m, cfg))


_nc(nlm::NLModel) = nlm.nc

ACE.params(nlm::NLModel) = svecs2vec(params(nlm.m))
ACE.nparams(nlm::NLModel) = ACE.nparams(nlm.m)


ACE.set_params!(nlm::NLModel, c::AbstractArray{T}) where T<:Number = ACE.set_params!(nlm.m, mat2svecs(c, nlm.nc))
ACE.set_params!(nlm::NLModel, c::AbstractVector{<: SVector{N, T}}) where {N, T}  = ACE.set_params!(nlm.m, c)

# This provides the standard interface of setter and getter functions to FluxPotentials

ACE.params(calc::FluxPotential) = params(calc.model)
ACE.nparams(calc::ACEflux.FluxPotential) = ACE.nparams(calc.model)
ACE.set_params!(calc::FluxPotential, c) = ACE.set_params!(calc.model, c)
_nc(calc::FluxPotential) = _nc(calc.model) # This function makes only sense if model is of type NLModel

# CombPotential provides a workaround to combine two different nonlinear model as a sum of two FluxPotentials
struct CombPotential <: SitePotential
    m1::FluxPotential # a function taking atoms and returning a site energy
    m2::FluxPotential
end

function JuLIP.energy(calc::CombPotential, at::Atoms)
    return energy(calc.m1,at) + energy(calc.m2,at)
end

function JuLIP.forces(calc::CombPotential, at::Atoms)
    return forces(calc.m1,at) + forces(calc.m2,at)
end

ACE.nparams(m::CombPotential) = nparams(m.m1) + nparams(m.m2)
ACE.params(m::CombPotential) = cat(params(m.m1),params(m.m2), dims=1)

    
function ACE.set_params!(m::CombPotential, c12::AbstractArray{T}) where T<:Number 
      c1, c2 = unpack(m, c12) 
      ACE.set_params!(m.m1, c1)
      ACE.set_params!(m.m2, c2)
      return m 
end
pack(::CombPotential, c1::AbstractArray{T}, c2::AbstractArray{T}) where T<:Number = cat(c1, c2, dims=1)
pack(::CombPotential, c1::AbstractVector{<: SVector{N, T}}, c2::AbstractVector{<: SVector{N, T}}) where {N, T} = cat(svecs2vec(c1), svecs2vec(c2), dims=1)
unpack(calc::CombPotential, c12::AbstractArray{T})  where T<:Number = mat2svecs(c12[1:(_nc(calc.m1)*ACE.nparams(calc.m1))],_nc(calc.m1)),  mat2svecs(c12[(_nc(calc.m2)*ACE.nparams(calc.m2)+1):end],_nc(calc.m2))



# FSModel(model1::ACE.LinearACEModel, model2::ACE.LinearACEModel, rcut::T) where {T<:Real} = FSModel(model1::ACE.LinearACEModel, model2::ACE.LinearACEModel, props ->  (1 .+ val.(props).^2).^0.5, rcut::T) 
# function JuLIP.energy(m::FSModel, at::Atoms) 
#     nlist = neighbourlist(at, m.rcut)
#     #E =  ACE.Invariant(0.0)
#     E =  SVector{1,Float64}(0.0)
#     for k = 1:length(at)
#         Js, Rs = NeighbourLists.neigs(nlist, k)    # Js = indices, Rs = PositionVectors 
#         cfg = ACEConfig( [ ACE.State(rr = r)  for r in Rs ] )
#         E += m.transform(evaluate(m.model2, cfg))
#         @show typeof(evaluate(m.model1, cfg))
#         E += val.(evaluate(m.model1, cfg))
#     end
#     return E
# end
# function JuLIP.energy(m::FSModel, at::Atoms) 
#     nlist = neighbourlist(at, m.rcut)
#     E =  0.0
#     for k = 1:length(at)
#         Js, Rs = NeighbourLists.neigs(nlist, k)    # Js = indices, Rs = PositionVectors 
#         cfg = ACEConfig( [ ACE.State(rr = r)  for r in Rs ] )
#         E += sum(m.transform(evaluate(m.model2, cfg)))
#         E += val(sum(evaluate(m.model1, cfg)))
#     end
#     return E
# end
# FSModel(model1::ACE.LinearACEModel, model2::ACE.LinearACEModel, rcut::T) where {T<:Real} = FSModel(model1::ACE.LinearACEModel, model2::ACE.LinearACEModel, props -> sum( (1 .+ val.(props).^2).^0.5 ), rcut::T) 

# function ACE.evaluate(m::FSModel, cfg::AbstractConfiguration) 
#     return evaluate(m.model1, cfg) + sum(m.transform(evaluate(m.model2, cfg)))
# end


# function JuLIP.energy(m::FSModel, at::Atoms) 
#     nlist = neighbourlist(at, m.rcut)
#     E =  0.0
#     for k = 1:length(at)
#         Js, Rs = NeighbourLists.neigs(nlist, k)    # Js = indices, Rs = PositionVectors 
#         cfg = ACEConfig( [ ACE.State(rr = r)  for r in Rs ] )
#         E += sum(m.transform(evaluate(m.model2, cfg)))
#         E += val(sum(evaluate(m.model1, cfg)))
#     end
#     return E
# end

# function JuLIP.forces(m::FSModel, at::Atoms) 
#     nlist = neighbourlist(at, m.rcut)
#     F = zeros(SVector{3,Float64}, length(at))
#     for k = 1:length(at)
#         Js, Rs = NeighbourLists.neigs(nlist, k)    # Js = indices, Rs = PositionVectors 
#         cfg = ACEConfig( [ ACE.State(rr = r)  for r in Rs ] )
#         g1 = Zygote.gradient(x ->  m.transform(evaluate(m.model2, x)), cfg)[1]
#         g2 = ACE.grad_config(m.model1, cfg)
#         for (i,j) in enumerate(Js)
#             F[j] += -g1[i].rr - g2[i][1].rr
#             F[k] +=  g1[i].rr + g2[i][1].rr
#         end
#     end
#     return F
# end


# function allocate_F(n::Int)
#     return zeros(SVector{3, Float64}, n)
# end

# ACE.nparams(m::FSModel) = ACE.nparams(m.model1) + ACE.nparams(m.model2)

# ACE.params(m::FSModel) = concat(copy(m.model1.c),copy(m.model2.c))

# function ACE.set_params!(m::FSModel, c) where T<:Number 
#     ACE.set_params!(m, c[1:nparams(m.model1)],:lin)
#     ACE.set_params!(m, c[(nparams(m.model1)+1):end], :nonlin)
#    return m 
# end

# function ACE.set_params!(m::FSModel, c, s::Symbol) where {T<:Number}
#     if s == :lin
#         ACE.set_params!(m.model1, c)
#     elseif s == :nonlin
#         ACE.set_params!(m.model2, c)
#     else
#         @error "non-valid reference to model in set_params "
#     end
#     return m
# end




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

end
