using ACE, JuLIP, ACEatoms, LinearAlgebra
using BayesianMLIP.NLModels
using Random
#using ProgressMeter
using PyPlot

struct CombModel <: AbstractCalculator
    model_list
end

function forces(model::CombModel, at::AbstractAtoms)
    return sum(forces(m,at) for m in model.model_list)
end

function energy(model::CombModel, at::AbstractAtoms)
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


model = CombModel([fsmodel, Vpair])


#%%
using BayesianMLIP: run!, BAOAB
at = bulk(:Al, cubic=true) * 3
N_obs = 10
h = 0.01
N = length(at)
ld = BAOAB(h, N; γ=1.0, β=1.0)
data = []
for k = 1:N_obs
    run!(ld, model, at, 1000)
    push!(data,(at=deepcopy(at),E = energy(fsmodel,at), F = forces(fsmodel,at) ))
end

# 1) Check that Hamiltonian dynamics preservers total energy of mores model. Confirmed
# 2) check that combined model works... 
# 3) generate data using the combined model



# Define Likelihood for isotropic Gaussian noise 
# Define prior 
