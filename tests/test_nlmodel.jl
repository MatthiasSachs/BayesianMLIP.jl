using ACE
using ACEatoms
using BayesianMLIP.NLModels
using Random: seed!, rand
using JuLIP

maxdeg = 4 # max degree
ord = 2 # max body order correlation 
Bsel = SimpleSparseBasis(ord, maxdeg)
rcut = 5.0 # Cutoff radius

# Create phi_mnl one particle basis
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, Bsel = Bsel, 
                                 rin = 1.2, rcut = 5.0)

# Create Symmetric bases for fmmodel
ACE.init1pspec!(B1p, Bsel)
basis1 = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
basis2 = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)

# Create bulk configuration
at = bulk(:Ti, cubic=true) * 3
rattle!(at,0.1)

#Create FM model
model = FSModel(basis1, basis2, 
                            rcut, 
                            x -> -sqrt(x), 
                            x -> 1 / (2 * sqrt(x)), 
                            ones(length(basis1)), ones(length(basis2)))
#Evaluate model at bulk configuration
E = ACE.evaluate(model, at)

println(typeof(at)) #Atoms{Float64}   