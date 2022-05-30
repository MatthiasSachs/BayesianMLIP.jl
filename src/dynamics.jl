using ACE

using JuLIP: AbstractAtoms
abstract type Dynamics end

"""
type "dynamics" must implement traverse!(at::AbstractAtoms; kwargs)
"""
#run(d::dynamics,at::AbstractAtoms, Nsteps::i)

abstract type HamiltonianDynamics <: Dynamics end

B_step!(d::HamiltonianDynamics, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_momenta!(at, at.P + hf * d.h * d.forces)
A_step!(d::HamiltonianDynamics, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_positions!(at,at.X + hf * d.h * at.P./at.M)



mutable struct VVerlet{T} <: HamiltonianDynamics where {T}
    at::AbstractAtoms
    force::Vector{JVec{T}} # force
    h::Float64      # step size
 end

 function step!(s::VelocityVerlet, V, at::AbstractAtoms ) #V::SitePotential
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=1.0)
    s.forces = forces(V, at)
    B_step!(s, at; hf=.5)
end

abstract type Thermostat <: HamiltonianDynamics end
"""
must implement get_invtemp
"""

get_invtemp(d::Thermostat) = d.β


abstract type Langevin <: Thermostat end 

O_step!(s::Langevin, at::AbstractAtoms) where {T<:Real} = set_momenta!(at, s.α .* at.P + s.ζ * randn(SVector{3,Float64},length(at)) )

mutable struct BAOAB{T}  <: Langevin where {T<:Real}
    h::T
    forces::Vector{SVector{3,T}}
    β::T
    α::T
    ζ::T
end

BAOAB(h::T, N::Int; γ::T=1.0, β::T=1.0) where {T<:Real} = BOAOB(h, zeros(SVector{3,T},N), β, exp(-h *γ ), sqrt( 1.0/β * (1-exp(-2*h*γ)))) 

function step!(s::BAOAB, V::SitePotential, at::AbstractAtoms )
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=.5)
    O_step!(s, at)
    A_step!(s, at; hf=.5)
    s.forces = forces(V, at)
    B_step!(s, at; hf=.5)
end