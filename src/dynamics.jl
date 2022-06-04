module Dynamics 


using ACE
using JuLIP: AbstractAtoms
using JuLIP
using StaticArrays
using Plots
using Random: seed!, rand
using LinearAlgebra: dot

include("outputschedulers.jl")
using .Outputschedulers
include("nlmodels.jl")
using .NLModels

export run!, step!, animate!
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
function run!(d::Integrator, V, at::AbstractAtoms, Nsteps::Int; outp = nothing)
    if outp === nothing 
        for _ in 1:Nsteps 
            step!(d::Integrator, V, at)
            # println(Hamiltonian(V, at))
        end 
    else 
        for _ in 1:Nsteps 
            step!(d::Integrator, V, at)
            push!(outp.X_traj, copy(at.X))
            push!(outp.P_traj, copy(at.P))
            println(Hamiltonian(V, at))
        end
    end 
end


# animation 

function animate!(outp ; name::String="anim", trace=false)
    anim = @animate for t in 1:length(outp.X_traj)
        frame = outp.X_traj[t]  # a no_of_particles-vector with each element 
        XYZ_Coords = [ [point[1] for point in frame], [point[2] for point in frame], [point[3] for point in frame] ]

        if trace == true 
            scatter!(XYZ_Coords[1], XYZ_Coords[2], XYZ_Coords[3], title="Trajectory", framestyle=:grid, marker=2, 
                    markercolor="black", legend=false)
        else 
            scatter(XYZ_Coords[1], XYZ_Coords[2], XYZ_Coords[3], title="Trajectory", framestyle=:grid, marker=2, 
                    markercolor="black", legend=false)
        end 
    end
    gif(anim, "$(name).mp4", fps=50)
end

function Hamiltonian(V, at::Atoms) 
    # Wish we would directly call this on outp, but this would require outp to store 
    # entire at object rather than at.X and at.P
    PE = energy(V, at)
    KE = 0.5 * sum([dot(at.P[t] /at.M[t], at.P[t]) for t in 1:length(at.P)])
    return PE + KE 
end 


function mainFinnisSinclairSimulation() 
    maxdeg = 4
    ord = 2
    Bsel = SimpleSparseBasis(ord, maxdeg)
    rcut = 5.0 
    
    B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, Bsel = Bsel, 
                                        rin = 1.2, rcut = 5.0)
    ACE.init1pspec!(B1p, Bsel)
    basis1 = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
    basis2 = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
    
    
    at = bulk(:Ti, cubic=true) * 3
    
    rattle!(at,0.1) 
    model = FSModel(basis1, basis2, 
                        rcut, 
                        x -> -sqrt(x), 
                        x -> 1 / (2 * sqrt(x)), 
                        ones(length(basis1)), ones(length(basis2)))
    
    # E = energy(model, at)   
    F = forces(model, at)
    # grad1, grad2 = FS_paramGrad(model, at)
    
    VVIntegrator = VelocityVerlet(F, 0.05)
    outp = simpleoutp()
    run!(VVIntegrator, model, at, 260; outp = outp)
    # for step in 1:250 
    #     println(": ", outp.X_traj[step][1])
    # end 
    
    animate!(outp, name="FS_Animation")
end 

function mainMorseSimulation() 
    @info("Define (Morse) pair-potential")
    r0 = rnn(:Al)
    Vpair = JuLIP.morse(;A=4.0, e0=.5, r0=r0, rcut=(1.9*r0, 2.7*r0))
    
    @info("Create random Al configuration")
    seed!(1234)
    at = bulk(:Al, cubic=true) * 3
    at = rattle!(at, 0.1)
    
    # F = forces(Vpair, at)
    # E = energy(Vpair, at)
    
    VVIntegrator = VelocityVerlet(0.1, Vpair, at)
    PVIntegrator = PositionVerlet(0.1, Vpair, at)
    BAOIntegrator = BAOAB(0.1, Vpair, at)
    outp = simpleoutp()
    Nsteps = 500
    run!(BAOIntegrator, Vpair, at, Nsteps; outp = outp)
    animate!(outp, name="Morse_Animation")
end 

#at.pbc = (false,false, false)
#sum(sum(-f .* x for (f, x) in zip(F[t], at.X[t]))/(3*N), t=1:Nsteps)/Nsteps ≈ 1/β        # Configurational temperature 

#sum(sum(at.X[t] .* F[t])/(3*N), t=1:Nsteps)/Nsteps ≈ 1/β        # Configurational temperature 
# As step size tends to 0, the difference between the two above should tend to 0


end  # end module 