using ACE 
using JuLIP: AbstractAtoms

# Abstract Types                                                           Concrete Types 
# Dynamics - HamiltonianDynamics ----------------------------------------- VelocityVerlet(at, F, h)
#                                ----------------------------------------- PositionVerlet(at, F, h)
#                                - Thermostat -- Langevin ---------------- BAOAB 
#                                                         ---------------- BOAOB
#                                                         ---------------- OBABO
#                                                         ---------------- ABOBA
#                                                         ---------------- OABAO
#          - GradientDynamics ---- OLDDynamics --------------------------- EulerMaruyama(at, h, β) - 

abstract type Dynamics end 
abstract type HamiltonianDynamics <: Dynamics end
abstract type Thermostat <: HamiltonianDynamics end
abstract type Langevin <: Thermostat end 

abstract type GradientDynamics <: Dynamics end
abstract type OLDDynamics <: GradientDynamics end

# Implement subfunctions
B_step!(d::HamiltonianDynamics, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_momenta!(at, at.P + hf * d.h * d.forces)
A_step!(d::HamiltonianDynamics, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_positions!(at,at.X + hf * d.h * at.P./at.M)
O_step!(s::Langevin, at::AbstractAtoms) = set_momenta!(at, s.α .* at.P + s.ζ * randn(SVector{3,Float64},length(at)) )


# Every integrator (Dynamics) struct has:
# Every step! function takes in: 
#       1. Integrator (Dynamics) struct 
#       2. A Site Potential model 
#       3. AbstractAtoms type 

# Velocity Verlet Implementation 
mutable struct VelocityVerlet{T} <: HamiltonianDynamics where {T}
    at::AbstractAtoms
    F::Vector{JVec{T}} # force
    h::Float64      # step size
end

function step!(s::VelocityVerlet, V, at::AbstractAtoms )
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

mutable struct EulerMaruyama{T} <: OLDDynamics where {T<:Real}
    at::AbstractAtoms
    h::T                # step size 
    β::T                # integration parameter
end

function step!(d::EulerMaruyama, V, at::AbstractAtoms; hf::T=1.0) where {T}
    set_positions!(at, at.X + hf * d.h * forces(V, at)./at.M + sqrt.( hf/d.β * d.h/at.M).*randn(SVector{3,Float64},length(at)))
end

get_invtemp(d::Thermostat) = d.β

mutable struct BAOAB{T} <: Langevin where {T<:Real}
    h::T        # Step size 
    forces::Vector{ACE.SVector{3,T}}
    β::T        # Inverse Temperature 
    α::T        # Integrator parameters
    ζ::T        # Integrator parameters 
end

function step!(s::BAOAB, V, at::AbstractAtoms)
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=.5)
    O_step!(s, at)
    A_step!(s, at; hf=.5)
    s.forces = forces(V, at)
    B_step!(s, at; hf=.5)
end

