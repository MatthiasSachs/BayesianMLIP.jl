
using ACE

using JuLIP: AbstractAtoms
abstract type Dynamics end

"""
type "dynamics" must implement traverse!(at::AbstractAtoms; kwargs)
"""
function run!(d::Dynamics, N::Int)
    for t = 1:N
        step!(d::Dynamics, V, d.at; hf::T=1.0)
    end
end
# run(d::dynamics,at::AbstractAtoms, Nsteps::i)
abstract type GradientDynamics <: Dynamics end

abstract type OLDDynamics <: GradientDynamics end

mutable struct EulerMaruyama{T} <: OLDDynamics where {T<:Real}
    at::AbstractAtoms
    h::T 
    β::T    # step size
 end

function step!(d::EulerMaruyama, V, at::AbstractAtoms; hf::T=1.0) where {T}
    set_positions!(at, at.X + hf * d.h * forces(V, at)./at.M + sqrt.( hf/d.β * d.h/at.M).*randn(SVector{3,Float64},length(at)))
end

abstract type HamiltonianDynamics <: Dynamics end

B_step!(d::HamiltonianDynamics, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_momenta!(at, at.P + hf * d.h * d.forces)
A_step!(d::HamiltonianDynamics, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_positions!(at,at.X + hf * d.h * at.P./at.M)


mutable struct VelocityVerlet{T} <: HamiltonianDynamics where {T}
    at::AbstractAtoms
    F::Vector{JVec{T}} # force
    h::Float64      # step size
 end

function step!(s::VelocityVerlet, V, at::AbstractAtoms ) #V::SitePotential
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=1.0)
    s.F = forces(V, at)
    B_step!(s, at; hf=.5)
end

mutable struct PositionVerlet{T} <: HamiltonianDynamics where {T}
    at::AbstractAtoms
    F::Vector{JVec{T}} # force
    h::Float64      # step size
 end

function step!(s::PositionVerlet, V, at::AbstractAtoms ) #V::SitePotential
    A_step!(s, at; hf=.5)
    s.F = forces(V, at)
    B_step!(s, at; hf=1.0)
    A_step!(s, at; hf=.5)
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
    forces::Vector{ACE.SVector{3,T}}
    β::T
    α::T
    ζ::T
end

BAOAB(h::T, N::Int; γ::T=1.0, β::T=1.0) where {T<:Real} = BOAOB(h, zeros(SVector{3,T},N), β, exp(-h *γ ), sqrt( 1.0/β * (1-exp(-2*h*γ))))  

function step!(s::BAOAB, V, at::AbstractAtoms )
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=.5)
    O_step!(s, at)
    A_step!(s, at; hf=.5)
    s.forces = forces(V, at)
    B_step!(s, at; hf=.5)
end

at.pbc = (false,false, false)
sum(sum(at.X[t] .* force[t])/(3*N), t=1:Nsteps)/Nsteps ≈ 1/β        # Configurational temperature 
# As step size tends to 0, the difference between the two above should tend to 0

