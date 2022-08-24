module Dynamics 

using ACE, JuLIP, StaticArrays
using JuLIP: AbstractAtoms
using Random: seed!, rand
using LinearAlgebra: dot
using BayesianMLIP.Outputschedulers
using BayesianMLIP.NLModels
import BayesianMLIP.NLModels: Hamiltonian, energy, forces

export run!, step!, Integrator 
export VelocityVerlet, PositionVerlet, EulerMaruyama, BAOAB, BADODAB


abstract type Integrator end
abstract type HamiltonianIntegrator <: Integrator end 
abstract type LangevinIntegrator <: Integrator end 
abstract type AdaptiveLangevinIntegrator <: Integrator end 


# Ordinary Hamiltonian Integrators & Step Functions

mutable struct VelocityVerlet{T} <: HamiltonianIntegrator where {T <: Real}
    F::Vector{ACE.SVector{3,T}} # force
    h::T                        # step size
end
VelocityVerlet(h::Float64, V, at::AbstractAtoms) = VelocityVerlet(forces(V, at), h)

mutable struct PositionVerlet{T} <: HamiltonianIntegrator where {T <: Real}
    F::Vector{ACE.SVector{3,T}} # force
    h::T                        # step size
end
PositionVerlet(h::Float64, V, at::AbstractAtoms) = PositionVerlet(Forces(V, at) , h) 

B_step!(d::HamiltonianIntegrator, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_momenta!(at, at.P + hf * d.h * d.F)
A_step!(d::HamiltonianIntegrator, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_positions!(at, at.X + hf * d.h * at.P./at.M)

function step!(s::VelocityVerlet, V, at::AbstractAtoms) #V::SitePotential (e.g. FSModel)
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=1.0)
    s.F = forces(V, at)
    B_step!(s, at; hf=.5)
end

function step!(s::PositionVerlet, V, at::AbstractAtoms) #V::SitePotential
    A_step!(s, at; hf=.5)
    s.F = Forces(V, at)
    B_step!(s, at; hf=1.0)
    A_step!(s, at; hf=.5) 
end


# Langevin Integrators

get_invtemp(d::LangevinIntegrator) = d.β     # Function to retrieve inverse temperature

mutable struct EulerMaruyama{T} <: LangevinIntegrator where {T <: Real}
    h::T    # step size
    β::T    # Inverse temperature
end
function step!(d::EulerMaruyama, V, at::AbstractAtoms; hf::T=2.0) where {T}
    set_positions!(at, at.X + hf * d.h * forces(V, at)./at.M + sqrt.( hf/d.β * d.h * (1 ./ at.M)).*randn(SVector{3,Float64},length(at)))
end

mutable struct BAOAB{T} <: LangevinIntegrator where {T <: Real}  
    h::T        # Step size 
    F::Vector{SVector{3,T}}           # Force vector
    β::T        # Inverse Temperature 
    α::T        # Integrator parameters
    ζ::T        # Integrator parameters 
end
BAOAB(h::T, V, at::AbstractAtoms; γ::T=1.0, β::T=1.0) where {T<:Real} = BAOAB(h, forces(V, at), β, exp(-h *γ ), sqrt( 1.0/β * (1-exp(-2*h*γ))))

B_step!(d::LangevinIntegrator, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_momenta!(at, at.P + hf * d.h * d.F)
A_step!(d::LangevinIntegrator, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_positions!(at, at.X + hf * d.h * at.P./at.M)
O_step!(d::LangevinIntegrator, at::AbstractAtoms) = set_momenta!(at, d.α .* at.P + d.ζ * sqrt.(at.M) .* randn(ACE.SVector{3,Float64}, length(at)) )


function step!(s::BAOAB, V, at::AbstractAtoms) 
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=.5)
    O_step!(s, at)
    A_step!(s, at; hf=.5)
    s.F = forces(V, at)
    B_step!(s, at; hf=.5)
end


# Adaptive Langevin Integrators

get_invtemp(d::AdaptiveLangevinIntegrator) = d.β     # Function to retrieve inverse temperature

mutable struct BADODAB{T} <: AdaptiveLangevinIntegrator where {T <: Real}
    h::T        # Step size
    F::Vector{SVector{3,T}}     # Force vector 
    β::T        # Inverse temperature 
    n::Int64    # Degrees of freedom 
    σG::T 
    σA::T 
    μ::T 
    ξ::T
end 

BADODAB(h::T, V, at::AbstractAtoms; β::T=1.0, n::Int64=3*length(at), σG::T=1.0, σA::T=9.0, μ::T=10.0, ξ::T=1.0) where {T<:Real} = BADODAB(h, forces(V, at), β, n, σG, σA, μ, ξ)


B_step!(d::AdaptiveLangevinIntegrator, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_momenta!(at, at.P + hf * d.h * (d.F + d.σG * sqrt.(at.M) .* randn(ACE.SVector{3,Float64},length(at))))
A_step!(d::AdaptiveLangevinIntegrator, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_positions!(at, at.X + hf * d.h * at.P./at.M)
function D_step!(d::AdaptiveLangevinIntegrator, at::AbstractAtoms; hf::T=1.0) where {T<:Real} 
    d.ξ += hf * d.h * (1/d.μ) * (dot(at.P, at.P./at.M) - d.n * (1/d.β))
end 


function step!(s::BADODAB, pot, at::AbstractAtoms) 
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=.5)
    D_step!(s, at; hf=.5)
    
    if -0.001 < s.ξ < 0.001  
        s.P = s.P + sqrt(s.h) * s.σA * at.M .* randn(ACE.SVector{3,Float64},length(at))
    else 
        α = 1 - exp(-2 * s.ξ * s.h) 
        ζ = 2 * s.ξ
        at.P = exp(-s.ξ * s.h) * at.P + s.σA * sqrt.((α/ζ) .* at.M) .* randn(ACE.SVector{3,Float64},length(at))
    end 

    D_step!(s, at; hf=.5)
    A_step!(s, at; hf=.5)
    s.F = forces(pot, at)
    B_step!(s, at; hf=.5)
end



# Run function
function run!(d::Integrator, V, at::Atoms, Nsteps::Int; outp = nothing)
    if outp === nothing 
        for i in 1:Nsteps 
            step!(d, V, at)
            println(i)
        end 
    else 
        # push!(outp, at)
        for i in 1:Nsteps 
            step!(d, V, at)
            push!(outp, deepcopy(at))
            println(i, "/", Nsteps)
        end
    end 
end

# Integrators with respect to Parameter θ




end  # end module 