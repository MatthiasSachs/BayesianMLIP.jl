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
# construct the basis

maxdeg = 4
ord = 2
Bsel = SimpleSparseBasis(ord, maxdeg)
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
φ = ACE.Invariant()
basis1 = SymmetricBasis(φ, B1p, O3(), Bsel)

maxdeg = 4
ord = 2
Bsel = SimpleSparseBasis(ord, maxdeg)
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
φ = ACE.Invariant()
basis2 = SymmetricBasis(φ, B1p, O3(), Bsel)

#Define generalized Finnis-Sinclair model
FS = props -> sum( (1 .+ val.(props).^2).^0.5 )
ID = props -> sum( val.(props) )
np = 1
c_m1 = rand(SVector{np,Float64}, length(basis1));
nlm1 = NLModel(ACE.LinearACEModel(basis, c_m1, evaluator = :standard), np, ID);
c_m2 = rand(SVector{np,Float64}, length(basis2));
nlm2 = NLModel(ACE.LinearACEModel(basis, c_m1, evaluator = :standard), np, FS);

rcut = 2*rnn(:Al)


model = CombPotential(FluxPotential(nlm1, rcut),FluxPotential(nlm2, rcut));# this is the FS-model

# test whether pack and unpack routines work as intended
c12 = pack(model,svecs2vec(c_m1),svecs2vec(c_m2))
c12b = pack(model,c_m1,c_m2)
@test c12 == c12b
c1, c2 = unpack(model, c12)
@test c_m1 == c1 && c2 == c_m2

# test energy and force evaluations
at = bulk(:Al, cubic=true) * 3 
rattle!(at,.5)
E1 = energy(model,at)
F1 = forces(model,at)



using Distributions
using Distributions: logpdf, MvNormal
using LinearAlgebra 
using BayesianMLIP.NLModels: params

Ndata = 100
data = []
for k = 1:Ndata
    data = push!(data,(at= rattle!(at,.1), E= randn(), F= randn(SVector{3, Float64}, length(F1))))
end

# Define log_likelihood function
w0 = 1.0 
weight_E, weight_F = w0, w0/ (3*length(at)) # weights correspond to precision of noise

log_likelihood = (model, d) -> -weight_E * (d.E - energy(model,d.at)) -  weight_F * sum(sum(abs2, g.rr - f) 
                     for (g, f) in zip(forces(model, d.at), d.F))
# Define prior 
prior = MvNormal(zeros(length(c12)),I)

# Define log posterior 
function log_posterior(model, data, prior, log_likelihood)
    return sum(log_likelihood(model, d) for d in data) + logpdf(prior, params(model))
end

# typtical steps to evaluate the log-posterior 
set_params!(model,randn(length(c12)));
log_posterior(model, data, prior, log_likelihood)

@time log_posterior(model, data, prior, log_likelihood) # Speed quite slow... Can we optimize this?





