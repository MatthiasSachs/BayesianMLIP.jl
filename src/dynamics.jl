module Dynamics 


using ACE
using JuLIP: AbstractAtoms
using JuLIP
using StaticArrays
using Plots
using Random: seed!, rand
using LinearAlgebra: dot

using BayesianMLIP.Outputschedulers
using BayesianMLIP.NLModels


export run!, step!
export VelocityVerlet, PositionVerlet, EulerMaruyama, BAOAB

# Consturct Hierarchy of Abstract Types for Organization
# Integrator --- HamiltonianIntegrator ------------------ VelocityVerlet(at, F, h)
#                                  ------------------ PositionVerlet(at, F, h)
#                                  ------------------ Thermostat ---------------- Langevin ----- BAOAB 
#          --- GradientIntegrator ----- OLDIntegrator --- EulerMaruyama(at, h, β)

abstract type Integrator end
abstract type HamiltonianIntegrator <: Integrator end 
abstract type GradientIntegrator <: Integrator end
abstract type OLDIntegrator <: GradientIntegrator end
abstract type Thermostat <: HamiltonianIntegrator end
abstract type Langevin <: Thermostat end 


# Ordinary Hamiltonian Integrators & Step Functions

mutable struct VelocityVerlet{T} <: HamiltonianIntegrator where {T}
    F::Vector{ACE.SVector{3,T}} # force
    h::Float64      # step size
end
VelocityVerlet(h::Float64, V, at::AbstractAtoms) = VelocityVerlet(forces(V, at), h)

mutable struct PositionVerlet{T} <: HamiltonianIntegrator where {T}
    F::Vector{ACE.SVector{3,T}} # force
    h::Float64      # step size
end
PositionVerlet(h::Float64, V, at::AbstractAtoms) = PositionVerlet(forces(V, at) , h) 

mutable struct EulerMaruyama{T} <: OLDIntegrator where {T<:Real}
    h::T 
    β::T    # step size
end

B_step!(d::HamiltonianIntegrator, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_momenta!(at, at.P + hf * d.h * d.F)
A_step!(d::HamiltonianIntegrator, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_positions!(at, at.X + hf * d.h * at.P./at.M)

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

function step!(d::EulerMaruyama, V, at::AbstractAtoms; hf::T=1.0) where {T}
    set_positions!(at, at.X + hf * d.h * forces(V, at)./at.M + sqrt.( hf/d.β * d.h/at.M).*randn(SVector{3,Float64},length(at)))
end


# Langevin Integrators 

get_invtemp(d::Thermostat) = d.β

mutable struct BAOAB{T} <: Langevin where {T<:Real}  
    h::T        # Step size 
    F::Vector{ACE.SVector{3,T}}
    β::T        # Inverse Temperature 
    α::T        # Integrator parameters
    ζ::T        # Integrator parameters 
end
BAOAB(h::T, V, at::AbstractAtoms ; γ::T=1.0, β::T=1.0) where {T<:Real} = BAOAB(h, forces(V, at), β, exp(-h *γ ), sqrt( 1.0/β * (1-exp(-2*h*γ))))

O_step!(s::Langevin, at::AbstractAtoms) = set_momenta!(at, s.α .* at.P + s.ζ * randn(ACE.SVector{3,Float64},length(at)) )

function step!(s::BAOAB, V, at::AbstractAtoms)
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=.5)
    O_step!(s, at)
    A_step!(s, at; hf=.5)
    s.F = forces(V, at)
    B_step!(s, at; hf=.5)
end

# Function that implements integrators over time interval. 
# Pushes information on at.X and at.P
function run!(d::Integrator, V, at::Atoms, Nsteps::Int; outp = nothing, config_temp = [])
    if outp === nothing 
        for _ in 1:Nsteps 
            step!(d, V, at)
            # println(Hamiltonian(V, at))
            # push!(config_temp, config_temperature(d.F, at.X))
        end 
    else 
        for _ in 1:Nsteps 
            step!(d, V, at)
            feed!(V, at, outp)
            # println(Hamiltonian(V, at))
            # push!(config_temp, config_temperature(d.F, at.X))
        end
    end 
end



#at.pbc = (false,false, false)
#sum(sum(-f .* x for (f, x) in zip(F[t], at.X[t]))/(3*N), t=1:Nsteps)/Nsteps ≈ 1/β        # Configurational temperature 

#sum(sum(at.X[t] .* F[t])/(3*N), t=1:Nsteps)/Nsteps ≈ 1/β        # Configurational temperature 
# As step size tends to 0, the difference between the two above should tend to 0


end  # end module 