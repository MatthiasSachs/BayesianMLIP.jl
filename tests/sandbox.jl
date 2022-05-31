using Revise
Revise.includet("../src/nlmodels.jl")
Revise.includet("../src/dynamics.jl")

using ACE
using ACEatoms
using BayesianMLIP.NLModels
using Random: seed!, rand
using JuLIP
using Plots
using BayesianMLIP.NLModels: eval_Model, eval_param_gradients, forces 

# Bayesian setting: Given a model with configurations, we estimate parameters for potential function 

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

E = eval_Model(model, at)
F = forces(model, at)
grad1, grad2 = eval_param_gradients(model, at)

using JuLIP: AbstractAtoms

# Consturct Hierarchy of Abstract Types for Organization
# Dynamics --- HamiltonianDynamics ------------------ VelocityVerlet(at, F, h)
#                                  ------------------ PositionVerlet(at, F, h)
#                                  ------------------ Thermostat ---------------- Langevin ----- BAOAB 
#          --- GradientDynamics ----- OLDDynamics --- EulerMaruyama(at, h, β)

abstract type Dynamics end
abstract type HamiltonianDynamics <: Dynamics end


mutable struct VelocityVerlet{T} <: HamiltonianDynamics where {T}
    at::AbstractAtoms
    F::Vector{JVec{T}} # force
    h::Float64      # step size
 end

mutable struct PositionVerlet{T} <: HamiltonianDynamics where {T}
    at::AbstractAtoms
    F::Vector{JVec{T}} # force
    h::Float64      # step size
 end

B_step!(d::HamiltonianDynamics, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_momenta!(at, at.P + hf * d.h * d.F)
A_step!(d::HamiltonianDynamics, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_positions!(at, at.X + hf * d.h * at.P./at.M)


# Multiple Dispatch of step! function dependeing on whether s is VelVer or PosVer 
function step!(s::VelocityVerlet, V, at::AbstractAtoms ) #V::SitePotential (e.g. FSModel)
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=1.0)
    s.F = forces(V, at)
    B_step!(s, at; hf=.5)
    potential_energy = eval_Model(V, at).val
    kinetic_energy = 0.5 * transpose(at.P./at.M) * at.P 
    Hamiltonian = potential_energy + kinetic_energy
    println("(PE, KE): ", potential_energy, "  ,  ", kinetic_energy)
    return s
end


function step!(s::PositionVerlet, V, at::AbstractAtoms ) #V::SitePotential
    A_step!(s, at; hf=.5)
    s.F = forces(V, at)
    B_step!(s, at; hf=1.0)
    A_step!(s, at; hf=.5)
    potential_energy = eval_Model(V, at).val
    kinetic_energy = 0.5 * transpose(at.P./at.M) * at.P 
    Hamiltonian = potential_energy + kinetic_energy
    println("(PE, KE): ", potential_energy, "  ,  ", kinetic_energy)
    return s
end

function run!(d::Dynamics, V, N::Int)
    for _ in 1:N
        step!(d, V, d.at)
    end
end

# animation 
function animate!(d::Dynamics, V, N::Int)
    anim = @animate for _ in 1:N
        try 
            step!(d, V, d.at)
            XYZ_Coords = [ [point[1] for point in d.at.X], [point[2] for point in d.at.X], [point[3] for point in d.at.X] ]
            scatter(XYZ_Coords[1], XYZ_Coords[2], XYZ_Coords[3])
        catch e
            print("Error happened")
            break 
        end
    end

    gif(anim, "anim.mp4", fps=50)
end


VVObj = VelocityVerlet(at, F, 0.01)
PVObj = PositionVerlet(at, F, 0.05) 
# step!(VVObj, model, at)

animate!(VVObj, model, 1500) 
animate!(PVObj, model, 500)


abstract type GradientDynamics <: Dynamics end

abstract type OLDDynamics <: GradientDynamics end

mutable struct EulerMaruyama{T} <: OLDDynamics where {T<:Real}
    at::AbstractAtoms
    h::T 
    β::T    # step size
 end

function step!(d::EulerMaruyama, V, at::AbstractAtoms) where {T}
    set_positions!(at, at.X + 1.0 * d.h * forces(V, at)./at.M + sqrt.( 1.0/d.β * d.h./at.M).*randn(ACE.SVector{3,Float64},length(at)))
    return d 
end

EMObj = EulerMaruyama(at, 0.05, 1.0)
step!(EMObj, model, at)
animate!(EMObj, model, 200)

#------------------------------------------------------

abstract type Thermostat <: HamiltonianDynamics end
"""
must implement get_invtemp
"""

get_invtemp(d::Thermostat) = d.β

abstract type Langevin <: Thermostat end 

O_step!(s::Langevin, at::AbstractAtoms) = set_momenta!(at, s.α .* at.P + s.ζ * randn(SVector{3,Float64},length(at)) )

mutable struct BAOAB{T} <: Langevin where {T<:Real}
    h::T        # Step size 
    forces::Vector{ACE.SVector{3,T}}
    β::T        # Inverse Temperature 
    α::T        # Integrator parameters
    ζ::T        # Integrator parameters 
end

BAOAB(h::T, N::Int; γ::T=1.0, β::T=1.0) where {T<:Real} = BOAOB(h, zeros(ACE.SVector{3,T},N), β, exp(-h *γ ), sqrt( 1.0/β * (1-exp(-2*h*γ))))  

BAOAB(1.0, 4)

function step!(s::BAOAB, V, at::AbstractAtoms)
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=.5)
    O_step!(s, at)
    A_step!(s, at; hf=.5)
    s.forces = forces(V, at)
    B_step!(s, at; hf=.5)
end

BAObj = BAOAB(0.1, F, 1.0, 1.0, 1.0)
get_invtemp(BAObj)

step!(BAObj, model, at)