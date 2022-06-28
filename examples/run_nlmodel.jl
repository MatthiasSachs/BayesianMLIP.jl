using ACE
using ACEatoms 
using BayesianMLIP           
using BayesianMLIP.NLModels         
using BayesianMLIP.Dynamics
using BayesianMLIP.Outputschedulers
using BayesianMLIP.Utils
using BayesianMLIP.MHoutputschedulers
using BayesianMLIP.Samplers 
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
using JLD2

function createFSmodel(maxdeg, ord)
    # construct two bases
    Bsel = SimpleSparseBasis(ord, maxdeg)
    B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
    φ = ACE.Invariant()
    basis1 = SymmetricBasis(φ, B1p, O3(), Bsel)

    Bsel = SimpleSparseBasis(ord, maxdeg)
    B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
    φ = ACE.Invariant()
    basis2 = SymmetricBasis(φ, B1p, O3(), Bsel)

    #Define generalized Finnis-Sinclair model
    FS = props -> sum( (1 .+ val.(props).^2).^0.5 )
    ID = props -> sum( val.(props) )
    np = 1
    c_m1 = rand(SVector{np,Float64}, length(basis1));
    nlm1 = NLModel(ACE.LinearACEModel(basis1, c_m1, evaluator = :standard), np, ID);
    c_m2 = rand(SVector{np,Float64}, length(basis2));
    nlm2 = NLModel(ACE.LinearACEModel(basis2, c_m2, evaluator = :standard), np, FS);

    rcut = 2*rnn(:Al)

    model = CombPotential(FluxPotential(nlm1, rcut),FluxPotential(nlm2, rcut));# this is the FS-model
    return model
end 

fsmodel = createFSmodel(6, 2);
true_θ = load_object("true_theta.jld2");
set_params!(fsmodel, true_θ);
save_object("true_theta.jld2", true_θ)
# k = load_object("true_theta.jld2")

at = bulk(:Al, cubic=true) * 2; 
rattle!(at,.5);

using Distributions
using Distributions: logpdf, MvNormal
using LinearAlgebra 
using BayesianMLIP.NLModels: params
using JLD2

sampler = BAOAB(0.001, fsmodel, at)

Ndata = 10
data = []
for k = 1:Ndata
    println(k)
    BayesianMLIP.Dynamics.run!(sampler, fsmodel, at, 1000; outp=nothing)
    push!(data,(at= deepcopy(at), E = energy(fsmodel, at), F= forces(fsmodel, at))) 

end

# save_object("data.jld2", data)
data = load_object("data.jld2")

# using JLD2
# save_object("data.jld2", data)
# k = load_object("data.jld2")

w0 = 1.0 
weight_E, weight_F = w0, w0/ (3*length(at)) # weights correspond to precision of noise

statModel = StatisticalModel(
    (model, d) -> -1.0 * (d.E - energy(model,d.at))^2, 
    MvNormal(zeros(length(true_θ)),I), 
    fsmodel, 
    data
);

# Want to maximize log_posterior, i.e. minimize U 
log_posterior(statModel, randn(length(true_θ)))
log_posterior(statModel, true_θ)        # true max 

logpdf(statModel.prior, true_θ)

function U(m::StatisticalModel, θ)
    return -log_posterior(m, θ)
end 


# Implement MH algorithm
using BayesianMLIP 
using BayesianMLIP.NLModels         
using BayesianMLIP.Dynamics   
using BayesianMLIP.Outputschedulers
using BayesianMLIP.Utils
using BayesianMLIP.MHoutputschedulers
using BayesianMLIP.Samplers 

outp = BayesianMLIP.Outputschedulers.MHoutp()     
outp = BayesianMLIP.MHoutputschedulers.MHoutp()    # error?

MetroHastings = BayesianMLIP.Samplers.SimpleMHsampler(true_θ)
MetroHastings = BayesianMLIP.Samplers.SimpleMHsampler(randn(length(true_θ)))
# BayesianMLIP.Samplers.step!(MetroHastings, statModel)
BayesianMLIP.Samplers.run!(MetroHastings, statModel, 1000, outp)

final = outp.θ_steps[length(outp.θ_steps)]

norm(final - θ_true)

length(outp.θ_steps)

logpdf(statModel.prior, final)

set_params!(statModel.model, c)
log_posterior(statModel, c) 


visual_x = []
visual_y = []

for elem in outp.θ_steps
    push!(visual_x, elem[1])
    push!(visual_y, elem[2])
end