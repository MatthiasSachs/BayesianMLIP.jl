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
using ACE: val, SymmetricBasis
using BayesianMLIP.NLModels: NLModel, CombPotential, svecs2vec, pack, unpack
using BayesianMLIP.ACEflux
using BayesianMLIP.ACEflux: FluxPotential

using Test

ord = 2
maxdeg = 4
Bsel = SimpleSparseBasis(ord, maxdeg)
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
φ = ACE.Invariant()
basis1 = SymmetricBasis(φ, B1p, O3(), Bsel)

#Define generalized Finnis-Sinclair model
FS = props -> sum( (1 .+ val.(props).^2).^0.5 )
ID = props -> sum( val.(props) )
np = 1
c_m1 = rand(SVector{np,Float64}, length(basis1));
nlm1 = FluxPotential(NLModel(ACE.LinearACEModel(basis1, c_m1, evaluator = :standard), np, ID), 5.0);;

at = bulk(:Al; cubic =true) *2 

using JuLIP: energy
#%%
compE = θ -> (set_params!(nlm1,θ); energy(nlm1,at) )

c_test = randn(length(params(nlm1)))
Zygote.gradient(compE,c_test)