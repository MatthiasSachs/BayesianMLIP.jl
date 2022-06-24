using JuLIP, StaticArrays
import ChainRulesCore, ChainRules
import ChainRulesCore: rrule, NoTangent
import ACE: PositionState, ACEConfig, DState
import JuLIP: energy, forces
using Zygote
using ACE
"""
The following code was copied with only minimal modifcations from https://github.com/ACEsuit/ACEflux.jl/blob/main/src/calculator.jl.
Authorship is with C. Ortner and A. Ross.

"""



# # ------------------------------------------------------------------------
# #    Define a Site Potential 
# # ------------------------------------------------------------------------

#define a SitePotential to enrich the model with a cutoff

mutable struct FluxPotential{TM, TC} <: SitePotential
   model::TM # a function taking atoms and returning a site energy
   cutoff::TC #in Angstroms
end

#define functions for the FluxPotential
(y::FluxPotential)(x) = y.model(x) 
NeighbourLists.cutoff(V::FluxPotential) = V.cutoff


# # ------------------------------------------------------------------------
# #   Functions to find the neighbours of an atom given a potential (a cutoff)
# # ------------------------------------------------------------------------

#functions we don't want to differentiate when calculating energies and forces
#this functions are differentiated explicitly in the rrules

function neighbours_R(calc::FluxPotential, at::Atoms)
   tmp = JuLIP.Potentials.alloc_temp(calc, at)
   domain=1:length(at)
   nlist = neighbourlist(at, cutoff(calc))

   TX = PositionState{Float64}
   TCFG = ACEConfig{TX}
   domain_R = TCFG[]
   for i in domain
      j, R, Z = JuLIP.Potentials.neigsz!(tmp, nlist, at, i)
      # cfg = ACEConfig(TX[ TX(rr = R[j]) for j in 1:length(R) ])
      cfg = ACEConfig( TX[ TX(rr = rr) for rr in R ] )
      push!(domain_R, cfg)
   end
   return domain_R
end

function ChainRules.rrule(::typeof(neighbours_R), calc::FluxPotential, at::Atoms)
   return neighbours_R(calc, at), dp -> (NoTangent(), dp, NoTangent())
end

function neighbours_J(calc::FluxPotential, at::Atoms)
   tmp = JuLIP.Potentials.alloc_temp(calc, at)
   domain=1:length(at)
   nlist = neighbourlist(at, cutoff(calc))
   J = []
   for i in domain
      j, R, Z = JuLIP.Potentials.neigsz!(tmp, nlist, at, i)
      push!(J, j)
   end
   return J
end

function ChainRules.rrule(::typeof(neighbours_J), calc::FluxPotential, at::Atoms)
   return neighbours_J(calc, at), dp -> (NoTangent(), dp, NoTangent())
end


# # ------------------------------------------------------------------------
# #    Functions to calculate energies and forces, as well as their derivatives
# # ------------------------------------------------------------------------

#Energy and forces calculators

function JuLIP.energy(calc::FluxPotential, at::Atoms)
   domain_R = neighbours_R(calc, at)
   return sum([calc(r) for r in domain_R])
end



function JuLIP.forces(calc::FluxPotential, at::Atoms)
   domain_R = neighbours_R(calc, at)
   J = neighbours_J(calc, at)
   return _eval_forces(calc, at, domain_R, J)
end

function _eval_forces(calc::FluxPotential, at::Atoms, domain_R, J)
   #frc = zeros(SVector{3, Float64}, length(at))
   frc = zeros(DState{NamedTuple{(:rr,), Tuple{SVector{3, Float64}}}},length(at))
   for (i,r) in enumerate(domain_R)
      # [1] local forces
      tmpfrc = Zygote.gradient(calc, r)[1]
      # [2] loc to glob
      frc += loc_to_glob(tmpfrc, J[i], length(at), i)
   end
   return frc
end

function loc_to_glob(Gi, Ji, Nat, i)
   #frc = zeros(SVector{3, Float64}, Nat)
   frc = zeros(ACE.DState{NamedTuple{(:rr,), Tuple{SVector{3, Float64}}}}, Nat)
   for a = 1:length(Ji)
      frc[Ji[a]] -= Gi[a]
      frc[i] += Gi[a]
   end
   return frc
end

# function _eval_forces(calc::FluxPotential, at::Atoms, domain_R, J)
#    frc = zeros(SVector{3, Float64}, length(at))
#    #frc = zeros(DState{NamedTuple{(:rr,), Tuple{SVector{3, Float64}}}},length(at))
#    for (i,r) in enumerate(domain_R)
#       # [1] local forces
#       tmpfrc = Zygote.gradient(calc, r)[1]
#       # [2] loc to glob
#       frc += loc_to_glob(tmpfrc, J[i], length(at), i)
#    end
#    return frc
# end

# function loc_to_glob(Gi, Ji, Nat, i)
#    frc = zeros(SVector{3, Float64}, Nat)
#    #frc = zeros(ACE.DState{NamedTuple{(:rr,), Tuple{SVector{3, Float64}}}}, Nat)
#    for a = 1:length(Ji)
#       frc[Ji[a]] -= Gi[a].rr
#       frc[i] += Gi[a].rr
#    end
#    return frc
# end

function rrule(::typeof(loc_to_glob), Gi, Ji, Nat, i)
   frc = loc_to_glob(Gi, Ji, Nat, i)
   
   function _pullback(dP)  #dp is the global dp 
      TDX = eltype(frc) # typeof( DState( rr = zero(SVector{3, Float64}) ) )
      dPi = zeros( TDX, length(Ji) )
      for a = 1:length(Ji)
         dPi[a] -= dP[Ji[a]]
         dPi[a] += dP[i]
      end
      return dPi
   end

   return frc, dP -> (NoTangent(), _pullback(dP), NoTangent(), NoTangent(), NoTangent())
end