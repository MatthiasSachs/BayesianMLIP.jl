
module Dynamics

using ACE

using JuLIP: AbstractAtoms
using JuLIP: JVec

using StaticArrays

export run!


abstract type Dynamics end
"""
type "Dynamics" must implement step!(at::AbstractAtoms; kwargs)
"""
abstract type HamiltonianDynamics <: Dynamics end

mutable struct VelocityVerlet{T} <: HamiltonianDynamics where {T}
    F::Vector{JVec{T}} # force
    h::Float64      # step size
end

VelocityVerlet(h::Float64, V, at::AbstractAtoms) = VelocityVerlet(forces(V, at), h)

mutable struct PositionVerlet{T} <: HamiltonianDynamics where {T}
    F::Vector{JVec{T}} # force
    h::Float64      # step size
end

PositionVerlet(h::Float64, V, at::AbstractAtoms) = PositionVerlet(forces(V, at) , h) 

B_step!(d::HamiltonianDynamics, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_momenta!(at, at.P + hf * d.h * d.F)
A_step!(d::HamiltonianDynamics, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_positions!(at, at.X + hf * d.h * at.P./at.M)



# outp = simpleoutp()
# run!(d::Dynamics, V, at::AbstractAtoms, Nsteps::Int; outputscheduler = outp)
# outp.x_traj

function run!(d::Dynamics, V, at::AbstractAtoms, Nsteps::Int; outputscheduler = outp)
    for t = 1:Nsteps
        step!(d::Dynamics, V, at)
        feed!(d,V, at, outp)
    end
end

abstract type outputscheduler end

begin struct simpleoutp
    x_traj
end
simpleoutp() = simpleoutp([])
function feed!(d,V, at, outp::simpleoutp)
    push!(outp.x_traj,at.X)
end

function step!(d::EulerMaruyama, V, at::AbstractAtoms; hf::T=1.0) where {T}
    set_positions!(at, at.X + hf * d.h * forces(V, at)./at.M + sqrt.( hf/d.β * d.h/at.M).*randn(SVector{3,Float64},length(at)))
end



function step!(s::VelocityVerlet, V, at::AbstractAtoms ) #V::SitePotential (e.g. FSModel)
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=1.0)
    s.F = forces(V, at)
    B_step!(s, at; hf=.5)
end

function step!(s::PositionVerlet, V, at::AbstractAtoms ) #V::SitePotential
    A_step!(s, at; hf=.5)
    s.F = forces(V, at)
    B_step!(s, at; hf=1.0)
    A_step!(s, at; hf=.5)
end

abstract type GradientDynamics <: Dynamics end

abstract type OLDDynamics <: GradientDynamics end

mutable struct EulerMaruyama{T} <: OLDDynamics where {T<:Real}
    h::T 
    β::T    # step size
 end

function step!(d::EulerMaruyama, V, at::AbstractAtoms) where {T}
    set_positions!(at, at.X + 1.0 * d.h * forces(V, at)./at.M + sqrt.( 1.0/d.β * d.h./at.M).*randn(ACE.SVector{3,Float64},length(at)))
end

abstract type Thermostat <: HamiltonianDynamics end

get_invtemp(d::Thermostat) = d.β

abstract type Langevin <: Thermostat end 

O_step!(s::Langevin, at::AbstractAtoms) = set_momenta!(at, s.α .* at.P + s.ζ * randn(ACE.SVector{3,Float64},length(at)) )

mutable struct BAOAB{T} <: Langevin where {T<:Real}  
    h::T        # Step size 
    F::Vector{ACE.SVector{3,T}}
    β::T        # Inverse Temperature 
    α::T        # Integrator parameters
    ζ::T        # Integrator parameters 

end

# Constructor for BAOAB struct 
BAOAB(h::T, V, at::AbstractAtoms ; γ::T=1.0, β::T=1.0) where {T<:Real} = BAOAB(h, forces(V, at), β, exp(-h *γ ), sqrt( 1.0/β * (1-exp(-2*h*γ))))

function step!(s::BAOAB, V, at::AbstractAtoms)
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=.5)
    O_step!(s, at)
    A_step!(s, at; hf=.5)
    s.F = forces(V, at)
    B_step!(s, at; hf=.5)
end

#at.pbc = (false,false, false)
#sum(sum(-f .* x for (f, x) in zip(F[t], at.X[t]))/(3*N), t=1:Nsteps)/Nsteps ≈ 1/β        # Configurational temperature 

#sum(sum(at.X[t] .* F[t])/(3*N), t=1:Nsteps)/Nsteps ≈ 1/β        # Configurational temperature 
# As step size tends to 0, the difference between the two above should tend to 0

end