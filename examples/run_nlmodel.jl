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

function createFSmodel(maxdeg::Int64, ord::Int64, rcut::Float64)
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

    model = CombPotential(FluxPotential(nlm1, rcut),FluxPotential(nlm2, rcut));# this is the FS-model
    return model
end 

fsmodel = createFSmodel(6, 2, 2*rnn(:Al));
θ = params(fsmodel); 
at = bulk(:Al, cubic=true) * 2; 
rattle!(at,.5);

using Distributions
using Distributions: logpdf, MvNormal, Normal 
using LinearAlgebra 
using BayesianMLIP.NLModels: params
using JLD2

sampler = BAOAB(0.001, fsmodel, at)

function generate_data(Ndata::Int64, sampler, model, at) 
    energy_std = 2.687172212269082      # 0.05 times the mean of 280 energies in dataset1
    forces_std = 0.012395149775211672   # 0.05 times the mean of the norm of 280*32*3 components of each energy in dataset1
    data = []   # data = Dict("theta" => theta, "data" => data)
    for k = 1:Ndata
        BayesianMLIP.Dynamics.run!(sampler, model, at, 1000; outp=nothing)
        # push data with Gaussian noise
        Energy = energy(fsmodel, at)
        Forces = forces(fsmodel, at) 
        NoisyEnergy = Energy + rand(Normal(0, energy_std))
        NoisyForces = [ f + ACE.DState(rr = forces_std * randn(SVector{3, Float64})) for f in Forces]
        push!( data, (at= deepcopy(at), E = NoisyEnergy, F= NoisyForces) )
        println("Data added: ", k)
    end

    theta = params(model)

    return (theta, data)
end 

Theta = load_object("Data/dataset1.jld2")["theta"];
Data = load_object("Data/dataset1.jld2")["data"];

w0 = 1.0 
weight_E, weight_F = w0, w0/ (3*length(at)) # weights correspond to precision of noise
log_likelihood = (model, d) -> -weight_E * abs(d.E - energy(model,d.at)) -  weight_F * sum(sum(abs2, g.rr - f.rr) 
                     for (g, f) in zip(forces(model, d.at), d.F))
log_likelihood_Energy = (model, d) -> -1.0 * (d.E - energy(model,d.at))^2

# Much of the computing power is used to solve for the force part of log_likelihood

priorNormal = MvNormal(zeros(length(params(fsmodel))),I)

statModel = StatisticalModel(
    log_likelihood_Energy, 
    priorNormal, 
    fsmodel, 
    Data
);

# Want to maximize log_posterior, i.e. minimize U 
log_posterior(statModel, randn(length(params(fsmodel))))
log_posterior(statModel, Theta)
# logpdf(statModel.prior, Theta)

function U(m::StatisticalModel, θ)
    return -log_posterior(m, θ)
end 

   
outp = BayesianMLIP.MHoutputschedulers.MHoutp()
# MetroHastings = BayesianMLIP.Samplers.SimpleMHsampler(true_θ)
MetroHastings = BayesianMLIP.Samplers.SimpleMHsampler(rand(length(params(fsmodel))))
BayesianMLIP.Samplers.run!(MetroHastings, statModel, 40, outp)

outp.log_posterior

timesteps = 500
no_trials = 30

outp_collection = [ BayesianMLIP.MHoutputschedulers.MHoutp() for i in 1:no_trials] 
for i in 1:no_trials 
    MetroHastings = BayesianMLIP.Samplers.SimpleMHsampler(rand(length(params(fsmodel))))
    BayesianMLIP.Samplers.run!(MetroHastings, statModel, timesteps, outp_collection[i])
end 

comb = log_posterior(statModel, Theta) * ones(timesteps)
for i in 1:no_trials
    comb = hcat(comb, outp_collection[i].log_posterior)
end 

plot(1:timesteps, comb, title="MH TimeStep vs Log Posterior Value", labels=nothing)




norm(final - Theta)

length(outp.θ_steps)

logpdf(statModel.prior, final)
