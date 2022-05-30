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
A_step!(d::HamiltonianDynamics, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_positions!(at,at.X + hf * d.h * at.P./at.M)


# Multiple Dispatch of step! function dependeing on whether s is VelVer or PosVer 
function step!(s::VelocityVerlet, V, at::AbstractAtoms ) #V::SitePotential
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=1.0)
    s.F = forces(V, at)
    B_step!(s, at; hf=.5)
    return s 
end

function step!(s::PositionVerlet, V, at::AbstractAtoms ) #V::SitePotential
    A_step!(s, at; hf=.5)
    s.F = forces(V, at)
    B_step!(s, at; hf=1.0)
    A_step!(s, at; hf=.5)
    return s
end


function run!(d::Dynamics, V, N::Int)
    for _ in 1:N
        step!(d, V, d.at)
    end
end

obj = VelocityVerlet(at, F, 0.1)
println(obj.at.X)
# step!(obj, model, at)

run!(obj, model, 10) 
println(obj.at.X)