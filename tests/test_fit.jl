using ACE, JuLIP, ACEatoms, LinearAlgebra
using BayesianMLIP.NLModels     # Error? What is the correct way to import local modules? 
using Random
#using ProgressMeter

# using PyPlot 

struct CombModel <: AbstractCalculator
    model_list
end

include("../src/nlmodels.jl")
using .NLModels
import .NLModels: forces, energy
include("../src/dynamics.jl")
using .Dynamics
include("../src/outputschedulers.jl")
using .Outputschedulers 
import .Outputschedulers: simpleoutp

function forces(model::CombModel, at::Atoms)
    return sum(forces(m,at) for m in model.model_list)
end

function energy(model::CombModel, at::Atoms)
    return sum(energy(m,at) for m in model.model_list)
end



maxdeg = 4 # max degree
ord = 2 # max body order correlation 
Bsel = SimpleSparseBasis(ord, maxdeg)
rcut = 2.7*rnn(:Al) # Cutoff radius

# Create phi_mnl one particle basis
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, Bsel = Bsel, 
                                 rin = 1.2, rcut = rcut)
# Create Symmetric bases for fmmodel
ACE.init1pspec!(B1p, Bsel)
basis1 = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
basis2 = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)

# Create bulk configuration
at = bulk(:Al, cubic=true) * 3      

rattle!(at,0.1)                     


#Create Finnis-Sinclair model, with struct defined in nlmodels.jl
fsmodel = FSModel(basis1, basis2, 
                    rcut, 
                    x -> -sqrt(x), 
                    x -> 1 / (2 * sqrt(x)), 
                    ones(length(basis1)), ones(length(basis2)))
r0 = rnn(:Al)
Vpair = JuLIP.morse(;A=4.0, e0=.5, r0=r0, rcut=(1.9*r0, rcut))


Cmodel = CombModel([fsmodel, Vpair])

VVIntegrator = VelocityVerlet(0.01, Cmodel, at)
outp = simpleoutp()
Nsteps = 1000
run!(VVIntegrator, Cmodel, at, Nsteps; outp = outp)
animate!(outp, name="CM_Animation", trace=false)




#%%

# Collecting data on at every 100 (collect_interval) steps
using BayesianMLIP: run!, BAOAB
at = bulk(:Al, cubic=true) * 3
N_obs = 5
h = 0.01
collect_interval = 100
ld = BAOAB(h, Cmodel, at; γ=1.0, β=1.0)
data = []
for k = 1:N_obs
    run!(ld, Cmodel, at, collect_interval)
    push!(data,(at=deepcopy(at), E = energy(Cmodel,at), F = forces(Cmodel,at) ))
end

println(length(data))
println(typeof(data[1])) 
println(data[1].E)
println(data[2].E)
println(data[3].E)
println(data[4].E)
println(data[5].E)

# 1) Check that Hamiltonian dynamics preservers total energy of mores model. Confirmed
# 2) check that combined model works...     Unfortunately is still not stable
# 3) generate data using the combined model. :: Done, stored in outp, also can see in the animation



# Define Likelihood for isotropic Gaussian noise 
# Define prior 
