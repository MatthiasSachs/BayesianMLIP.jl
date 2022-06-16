using ACE
using ACEatoms 
using BayesianMLIP           
using BayesianMLIP.NLModels         
using BayesianMLIP.Dynamics
using BayesianMLIP.Outputschedulers
using BayesianMLIP.Utils
using Random: seed!, rand
using JuLIP
using Plots 
using ACE: O3, evaluate
using StaticArrays
using ACE: val
#using BayesianMLIP.NLModels: set_params!
##


# construct the basis
maxdeg = 6
ord = 3
rcut = 2*rnn(:Al)
Bsel = SimpleSparseBasis(ord, maxdeg)
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, rcut=rcut)
φ = ACE.Invariant()
basis = ACE.SymmetricBasis(φ, B1p, O3(), Bsel)

#initialize the model
c1 = rand(SVector{1,Float64}, length(basis))
c2 = rand(SVector{1,Float64}, length(basis))
model1 = ACE.LinearACEModel(basis, c1, evaluator = :standard);
model2 = ACE.LinearACEModel(basis, c2, evaluator = :standard);

#use default transformation
m1 = FSModel(model1,model2, rcut);

#use costume transformation
FS = props -> sum( (1 .+ val.(props).^2).^0.5 )
m2 = FSModel(model1,model2, FS, rcut)


at = bulk(:Al, cubic=true) * 3 

E1 = energy(m1,at)
F1 = forces(m1,at)

E2 = energy(m2,at)
F2 = forces(m2,at);

rattle!(at,1.1)
sampler = VelocityVerlet(0.01, m1, at) 
outp = atoutp()
run!(sampler, m1, at, 1000; outp=outp)

plot(outp.energy, label="Potential Energy")
plot!(outp.kenergy, label="Kinetic Energy")
plot!(outp.hamiltonian, label="Hamiltonian")
